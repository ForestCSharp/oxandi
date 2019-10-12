use specs::prelude::*;

use crate::DeltaTime;

#[derive(Debug, Component, Deref, DerefMut)]
pub struct Position(pub glm::Vec3);

#[derive(Debug, Component, Deref, DerefMut)]
pub struct Velocity(pub glm::Vec3);

#[derive(Debug, Component, Deref, DerefMut)]
pub struct Acceleration(pub glm::Vec3);

//TODO: automatic numeric operators via some proc macro (see newtype_derive (created before proc macros so a bit odd))

#[derive(Component, Default)]
pub struct Printer(String);

pub struct PrinterSystem;

impl<'a> System<'a> for PrinterSystem {
    type SystemData = (ReadStorage<'a,Printer>);

    fn run(&mut self, (printers) : Self::SystemData) {
        (&printers)
        .join()
        .for_each(|printer| {
            println!("{}", printer.0);
        });
    }
}

pub struct VelocitySystem;

impl<'a> System<'a> for VelocitySystem {

    type SystemData = (Read<'a, DeltaTime>, WriteStorage<'a, Position>, ReadStorage<'a, Velocity>, Entities<'a>, WriteStorage<'a, Printer>);

    fn run(&mut self, (dt, mut positions, velocities, entities, mut printers): Self::SystemData) {
 
        //Testing inserting entities
        let new_entity = entities.create();
        printers.insert(new_entity, Printer(format!("printer: {}", printers.count())));

        (&velocities, &mut positions)
        .par_join()
        .for_each(|(vel, pos)| {
            pos.0 += vel.0 * dt.0;

            println!("New Pos: {} {} {}", pos.x, pos.y, pos.z);
        });
    }
}

pub struct AccelerationSystem;

impl<'a> System<'a> for AccelerationSystem {

    type SystemData = (Read<'a, DeltaTime>, WriteStorage<'a, Velocity>, ReadStorage<'a, Acceleration>);

    fn run(&mut self, (dt, mut vel, acc): Self::SystemData) {
        (&mut vel, &acc)
        .par_join()
        .for_each(|(vel, acc)| {
            vel.0 += acc.0 * dt.0;

            println!("New Vel: {} {} {}", vel.x, vel.y, vel.z);
        });
    }
}