#![allow(clippy::needless_return)]

extern crate rand;
use rand::Rng;

extern crate kdtree;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;

use rayon::prelude::*;

use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

fn random_points(num_points: usize, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Vec<Point> {
    let mut rng = rand::thread_rng();
    let mut reference_points = Vec::new();

    for _ in 0..num_points {
        reference_points.push(Point {
            x: rng.gen_range(x_min, x_max),
            y: rng.gen_range(y_min, y_max),
        });
    }
    return reference_points;
}

fn get_distance_squared(p1: &Point, p2: &Point) -> f64 {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    return dx * dx + dy * dy;
}

fn nearest_neighbor_noddy(point: &Point, neighbors: &[Point]) -> usize {
    let mut index = 0;
    let mut d_min = std::f64::MAX;

    for (i, neighbor) in neighbors.iter().enumerate() {
        let d = get_distance_squared(&point, &neighbor);
        if d < d_min {
            d_min = d;
            index = i;
        }
    }

    return index;
}

fn build_tree(neighbors: &[Point]) -> KdTree<f64, usize, [f64; 2]> {
    let dimensions = 2;

    let mut kdtree = KdTree::new(dimensions);

    for (index, neighbor) in neighbors.iter().enumerate() {
        kdtree.add([neighbor.x, neighbor.y], index).unwrap();
    }

    return kdtree;
}

fn nearest_index(point: &Point, kdtree: &KdTree<f64, usize, [f64; 2]>) -> usize {
    let result = kdtree
        .nearest(&[point.x, point.y], 1, &squared_euclidean)
        .unwrap();
    return *result[0].1;
}

fn main() {
    let (x_min, x_max) = (-10.0, 10.0);
    let (y_min, y_max) = (-10.0, 10.0);

    let num_neighbors = 50_000;
    let num_points = num_neighbors;

    let neighbors = random_points(num_neighbors, x_min, x_max, y_min, y_max);
    let points = random_points(num_points, x_min, x_max, y_min, y_max);

    let start = Instant::now();
    let kdtree = build_tree(&neighbors);
    println!("time elapsed in building tree: {:?}", start.elapsed());

    let start = Instant::now();
    let indices: Vec<usize> = points
        .par_iter()
        .map(|p| nearest_index(&p, &kdtree))
        .collect();
    println!("time elapsed in finding neighbors: {:?}", start.elapsed());

    let start = Instant::now();
    let indices_noddy: Vec<usize> = points
        .iter()
        .map(|p| nearest_neighbor_noddy(&p, &neighbors))
        .collect();
    println!("time elapsed in noddy: {:?}", start.elapsed());

    assert_eq!(indices, indices_noddy);
}
