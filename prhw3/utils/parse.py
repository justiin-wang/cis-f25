import numpy as np
import pandas as pd

def parse_rigid_bodies(path):
  with open(path, 'r') as f:
    data = f.readlines()
  num_trackers = int(data[0].split()[0])
  body_markers_bX = []
  for line in data[1:1+num_trackers]:
      coords = [float(x) for x in line.split()]
      body_markers_bX.append(coords)
  body_tip_bX = [float(x) for x in data[1+num_trackers].split()]
  return body_markers_bX, body_tip_bX, num_trackers

def parse_mesh(path):
  with open(path, 'r') as f:
    data = f.readlines()
  num_vertices = int(data[0])
  vertices_ct = []
  for line in data[1:1+num_vertices]:
    coords = [float(x) for x in line.split()]
    vertices_ct.append(coords)
  num_triangles = int(data[1+num_vertices])

  vertices_inds = []
  for line in data[2+num_vertices:2+num_vertices+num_triangles]:
    coords = [int(x) for x in line.split()]
    vertices_inds.append(coords[0:3])
  return vertices_ct, vertices_inds

def parse_readings(path, num_trackers_bA, num_trackers_bB):
  with open(path, 'r') as f:
    data = f.readlines()
  header = data[0].split(',')
  num_total_LEDs = int(header[0])
  num_dummy_LEDs = num_total_LEDs - num_trackers_bA - num_trackers_bB  # what is even the point???
  num_sample_frames = int(header[1])

  body_A_markers_tr = []
  body_B_markers_tr = []
  for i in range(num_sample_frames):
    for line in data[ 1 + i * num_total_LEDs : 1 + i * num_total_LEDs + num_trackers_bA]:
      coords = [float(x) for x in line.split(',')]
      body_A_markers_tr.append(coords[0:3])
    for line in data[1 + i * num_total_LEDs + num_trackers_bA : 1 + i * num_total_LEDs + num_trackers_bA + num_trackers_bB]:
      coords = [float(x) for x in line.split(',')]
      body_B_markers_tr.append(coords[0:3])
  return body_A_markers_tr, body_B_markers_tr, num_sample_frames

def parse_output(path):
    with open(path, 'r') as f:
        data = f.readlines()
    header = data[0].split(' ') # For some reason the header delim in the given outputs is a space
    num_samples = int(header[0].strip()) 

    dk = []
    ck = []
    for line in data[1:1 + num_samples]:
        vals = [float(x) for x in line.split()]
        if len(vals) < 6:
            raise ValueError(f"Strange input at line: {line}")
        dk.append(vals[0:3])
        ck.append(vals[3:6])

    return np.array(dk), np.array(ck)
  
    