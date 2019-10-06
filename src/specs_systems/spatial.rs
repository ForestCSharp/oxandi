use specs::prelude::*;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Component)]
pub struct Vel {
    pub value : glm::Vec3,
}

impl Deref for Vel {
    type Target = glm::Vec3;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl DerefMut for Vel {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

#[derive(Debug, Component)]
pub struct Pos {
    pub value : glm::Vec3,
}

impl Deref for Pos {
    type Target = glm::Vec3;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl DerefMut for Pos {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

pub struct MovementSystem;

impl<'a> System<'a> for MovementSystem {
    // These are the resources required for execution.
    // You can also define a struct and `#[derive(SystemData)]`,
    // see the `full` example.
    type SystemData = (WriteStorage<'a, Pos>, ReadStorage<'a, Vel>);

    fn run(&mut self, (mut pos, vel): Self::SystemData) {
        // The `.join()` combines multiple components,
        // so we only access those entities which have
        // both of them.
        // You could also use `par_join()` to get a rayon `ParallelIterator`.
        (&vel, &mut pos)
        .par_join()
        .for_each(|(vel, pos)| {
            pos.x += vel.x * 0.05;
            pos.y += vel.y * 0.05;

            println!("{}, {}", pos.x, pos.y);
        });
    }
}