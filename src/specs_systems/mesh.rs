use specs::prelude::*;

use rendy::{
    hal,
    factory::{Config, Factory},
    memory::Dynamic,
    mesh::PosColor,
    resource::{Escape, Buffer, BufferInfo},
};

use crate::gltf_loader::*;
use crate::SHADER_REFLECTION;

#[derive(Component)]
pub struct MeshComponent<B: hal::Backend> {
    pub vertex_buffer: Escape<Buffer<B>>,
    pub index_buffer : Escape<Buffer<B>>,
    pub num_indices  : u32,
    pub gltf_model : GltfModel,
}

impl<B> MeshComponent<B> where B: hal::Backend {
    pub fn new(factory: &mut Factory<B>) -> MeshComponent<B> {
        let cube = GltfModel::new("data/models/Cube.glb");
        let cube_vertices : Vec<PosColor> = cube.meshes[0].vertices.iter().map(|v| PosColor {
            position : v.a_pos.into(),
            color    : [1.0, 0.0, 0.0, 1.0].into(),
        }).collect();
        let cube_indices = &cube.meshes[0].indices.as_ref().unwrap();

        let vbuf_size = SHADER_REFLECTION.attributes_range(..).unwrap().stride as u64 * cube_vertices.len() as u64;

        let mut vbuf = factory
            .create_buffer(
                BufferInfo {
                    size: vbuf_size,
                    usage: hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            ).unwrap();

        unsafe {
        // Fresh buffer.
        factory
            .upload_visible_buffer(
                &mut vbuf,
                0,
                &cube_vertices,
            ).unwrap();
        }

        let ibuf_size = (std::mem::size_of::<usize>() * cube_indices.len()) as u64;

        let mut ibuf = factory
            .create_buffer(
                BufferInfo {
                    size  : ibuf_size,
                    usage : hal::buffer::Usage::INDEX,
                },
                Dynamic,
            ).unwrap();

        unsafe {
        factory
            .upload_visible_buffer(
                &mut ibuf,
                0,
                &cube_indices,
            ).unwrap();
        }

        MeshComponent {
            vertex_buffer : vbuf,
            index_buffer  : ibuf,
            num_indices   : cube_indices.len() as u32,
            gltf_model    : cube,
        }
    }
}