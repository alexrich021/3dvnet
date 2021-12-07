import numpy as np
import pyrender
import trimesh


class Renderer():
    """OpenGL mesh renderer

    Used to render depthmaps from a mesh for 2d evaluation
    """

    def __init__(self, mesh_o3d, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.height = height
        self.width = width
        self.scene = pyrender.Scene()
        self.mesh = self.mesh_o3d_to_opengl(mesh_o3d)
        self.scene.add(self.mesh)

    def __call__(self, intrinsics, pose):
        self.renderer.viewport_height = self.height
        self.renderer.viewport_width = self.width
        self.scene.clear()
        self.scene.add(self.mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(np.linalg.inv(pose)))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R =  np.array([[1, 0, 0],
                       [0, c, -s],
                       [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose@axis_transform

    def mesh_o3d_to_opengl(self, mesh):
        mesh_trimesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
        material = pyrender.material.Material(doubleSided=True)
        return pyrender.Mesh.from_trimesh(mesh_trimesh, material=material)

    def delete(self):
        self.renderer.delete()


def process_scene(mesh, poses, K, render_size=(480, 640)):
    renderer = Renderer(mesh, *render_size)
    depths = np.empty((poses.shape[0], *render_size))
    for i in range(poses.shape[0]):
        _, depth = renderer(K[i], poses[i])
        depths[i] = depth
    renderer.delete()
    return depths
