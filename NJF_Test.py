#Write a basic test with differentiable rendering to indicate the NJF use as a trainable parameters
import os.path
from enum import Enum
from functools import reduce
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from loguru import logger
from pytorch3d.io import load_obj, save_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, \
    SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection
from torch.utils.data import DataLoader
from pytorch3d.ops import sample_points_from_meshes

from NeuralJacobianFields import SourceMesh
from video_log import Video
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

# 3d loss
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)


# Two Head Meshes
TARGET_MESH_OBJ = 'Models/head_template_mesh_upsample_mouthopen.obj'
SOURCE_MESH_OBJ = 'Models/head_template_mesh_upsample.obj'


class NJF_Tester():
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.save_folder = './output/'
        self.actor_name = "head";
        self.video_folder = os.path.join(self.save_folder, self.actor_name, "video")
        self.mesh_folder = os.path.join(self.save_folder, self.actor_name, "mesh")

        # utilize checkpoint for easy setup for cameras
        self.frameid = 0;
        self.checkpoint_folder = os.path.join(self.save_folder, self.actor_name, "checkpoint")
        self.create_output_folders()
        self.initialize_mesh();


    def create_output_folders(self):
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_folder).mkdir(parents=True, exist_ok=True)
        Path(self.mesh_folder).mkdir(parents=True, exist_ok=True)
        Path(self.video_folder).mkdir(parents=True, exist_ok=True)
   
    def plot_pointcloud(self, mesh, title=""):
        # Sample points uniformly from the surface of the mesh.
        points = sample_points_from_meshes(mesh, 5000)
        x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(x, z, -y)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title(title)
        ax.view_init(190, 30)
        plt.savefig(os.path.join(self.video_folder, title + ".png"))

    def initialize_mesh(self):
        self.source_vertices, source_faces_group, _ = load_obj(SOURCE_MESH_OBJ)
        self.target_vertices, target_faces_group, _ = load_obj(TARGET_MESH_OBJ)

        self.source_faces_idx = source_faces_group.verts_idx.numpy();
        self.target_faces_idx = target_faces_group.verts_idx.to(self.device)
        self.source_vertices = self.source_vertices.numpy();
        self.target_vertices = self.target_vertices.to(self.device)   # group to gpu

        self.trg_mesh = Meshes(verts=[self.target_vertices], faces=[self.target_faces_idx])
       
        #set up jacobian
        jacobian_source = SourceMesh.SourceMesh(0, SOURCE_MESH_OBJ, {}, 1, ttype=torch.float,
                                                use_wks=False)    
        
        # need cpu and array
        jacobian_source.load(self.source_vertices, self.source_faces_idx);   #build NJF from vert and faces
        jacobian_source.to(self.device)
        
        #make it backto tensor...
        self.source_vertices = torch.from_numpy(self.source_vertices).to(self.device)
        self.source_faces_idx = torch.from_numpy(self.source_faces_idx).to(self.device)
        gt_jacobian = jacobian_source.jacobians_from_vertices(self.source_vertices.unsqueeze(0))   
        gt_jacobian.requires_grad = True;

        # put jacobian as a trainable parameter
        if self.load_checkpoint() is False:
            self.gt_jacobian = gt_jacobian;  # assign the new Jacobian and save the jacobian_source
        
        self.jacobian_source = jacobian_source;
        self.optimizer = torch.optim.SGD([self.gt_jacobian], lr=1.0, momentum=0.9)
        

    def load_checkpoint(self, idx=-1):
        print("Load Checkpoint from the Folder: ", self.checkpoint_folder);
        if not os.path.exists(self.checkpoint_folder):
            return False
        snaps = sorted(glob(self.checkpoint_folder + '/*.frame'))  # find all the checkpoint files
        if len(snaps) == 0:
            logger.info('Training from beginning...')
            return False

        last_snap = snaps[idx]
        payload = torch.load(last_snap)

        self.frameid = int(payload['frame_id']);  
        self.gt_jacobian = payload['jacobian'].to(self.device)
    
        logger.info(f'Snapshot loaded for frame {self.frameid+1}')
        self.frameid =self.frameid+1; #for next record

        return True

    def save_checkpoint(self, frame_id=0):
        frame = {
            'jacobian': self.gt_jacobian,
            'frame_id': frame_id,
        }
        
        torch.save(frame, f'{self.checkpoint_folder}/{frame_id}.frame')

    def run(self):
        # perform loss on the vertices directly
        # Number of optimization steps
        Niter = 500
        # Weight for the chamfer loss
        w_chamfer = 1.0 
        # Weight for mesh edge loss
        w_edge = 1.0 
        # Weight for mesh normal consistency
        w_normal = 1   # may need to enlarge this 
        # Weight for mesh laplacian smoothing
        w_laplacian = 0.1 
        # Plot period for the losses
        plot_period = 250
        loop = tqdm(range(Niter))

        chamfer_losses = []
        laplacian_losses = []
        edge_losses = []
        normal_losses = []

        
        for i in loop:
            # Initialize optimizer
            self.optimizer.zero_grad()
            # Deform the mesh
            deform_verts = self.jacobian_source.vertices_from_jacobians(self.gt_jacobian)
            new_src_mesh = Meshes(verts=[deform_verts.squeeze()], faces=[self.source_faces_idx])
            
            # We sample 5k points from the surface of each mesh 
            sample_trg = sample_points_from_meshes(self.trg_mesh, 15000)
            sample_src = sample_points_from_meshes(new_src_mesh, 15000)
            
            # We compare the two sets of pointclouds by computing (a) the chamfer loss
            loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
            
            # and (b) the edge length of the predicted mesh
            loss_edge = mesh_edge_loss(new_src_mesh)
            
            # mesh normal consistency
            loss_normal = mesh_normal_consistency(new_src_mesh)
            
            # mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
            
            # Weighted sum of the losses
            loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
            
            # Print the losses
            loop.set_description('total_loss = %.6f' % loss)
            
            # Save the losses for plotting
            chamfer_losses.append(float(loss_chamfer.detach().cpu()))
            edge_losses.append(float(loss_edge.detach().cpu()))
            normal_losses.append(float(loss_normal.detach().cpu()))
            laplacian_losses.append(float(loss_laplacian.detach().cpu()))
            
            # Plot mesh
            if i % plot_period == 0:
                self.plot_pointcloud(new_src_mesh, title="iter: %d" % i)
                
            # Optimization step
            loss.backward()
            self.optimizer.step()
        
        #save the predicted mesh
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        str_frameid = "predicted_mesh_" + str(self.frameid)+".obj";
        save_obj(os.path.join(self.mesh_folder, str_frameid), final_verts, final_faces)
        self.save_checkpoint(frame_id=self.frameid);

if __name__ == '__main__':
    ff = NJF_Tester(device='cuda:0')
    ff.run()