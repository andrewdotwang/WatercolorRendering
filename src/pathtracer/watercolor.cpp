#include "watercolor.h"

#include "scene/light.h"
#include "scene/sphere.h"
#include "scene/triangle.h"
//#include "scene/gl_scene/mesh.h"


using namespace CGL::SceneObjects;

namespace CGL {

WaterColor::WaterColor() {
  //init stuff or set params
  //tm_gamma = 2.2f;
}

WaterColor::~WaterColor() {
  //delete stuff if necessary
}

StatsBuilder::StatsBuilder() {
  sum = 0; 
}

void StatsBuilder::add(float n) {
  nums.push_back(n);
  sum += n;
}

Stats StatsBuilder::calc_stats() {
  int n = nums.size();
  if (n == 0) {
    return Stats();
  }
  float mean = sum / (float(n));
  float sum_sq_diff = 0.;
  float min = nums[0];
  float max = nums[0];
  for (float f: nums) {
    sum_sq_diff += (f - mean) * (f - mean);
    if (f < min) {
      min = f;
    }
    if (f > max) {
      max = f;
    }
  }
  float std = sqrt(sum_sq_diff/ float(n));

  return Stats(mean, std, min, max, n);
}

void StatsBuilder::clear() {
  sum = 0.;
  nums.clear();
}

void StatsBuilder::print_stats() {
  Stats s = calc_stats();
  cout << "mean: " << s.mean << " std: " << s.std << " min: " << s.min << " max: " << s.max << " n: " << s.num_vals << endl;
}

// dispatches to the appropriate method for simulating watercolor dispersion
// based on the type of GLScene::SceneObject
void WaterColor::simulate(GLScene::SceneObject *elem) {
  if (dynamic_cast<GLScene::Mesh*>(elem)) {
    simulate_mesh((GLScene::Mesh*)elem);
  }
}

bool WaterColor::can_see(FaceIter f) {
  Ray r = camera->generate_ray(0.5, 0.5);
  return dot(f->normal(), r.d) < -0.25;
}

/*
Returns a "random" face from the given mesh.

Note that our pRNG is comes from util/random_util.h
it doesn't even let you seed... so you'll get the same faces every time.
if you want more random results, write your own random_uniform function
using rand() and srand(time(NULL)) from the <random> library.
*/
FaceIter WaterColor::get_random_face(HalfedgeMesh& mesh) {
  FaceIter starter_f;
  unsigned long start_ind;


  start_ind = (unsigned long)(random_uniform() * mesh.nFaces());
  starter_f = mesh.facesBegin();
  for(int tmp = 0; tmp < start_ind; tmp++) {
    starter_f++;
  }
  return starter_f;
}

/* a nice method used to get a patch of faces of a given size on the mesh.

\param mesh The HalfedgeMesh from which we will return a set of touching faces
\param num_faces The number of faces that should be in the returned patch
\param visible if True, this indicates that the patch should be somewhere visible
               to the camera.
\return a list of Faces of the mesh that ccorrespond to a circle around a random ccenter point.
*/
std::vector<FaceIter> WaterColor::get_patch(HalfedgeMesh& mesh, int num_faces, bool visible) {
  std::unordered_set<Face*> all_seen;

  //start with random face

  FaceIter starter_f = get_random_face(mesh);

  if (visible) {
    while(!can_see(starter_f)) {
      starter_f = get_random_face(mesh);
    }
  }

  all_seen.insert(elementAddress(starter_f));
  std::queue<FaceIter> proc_q;
  proc_q.push(starter_f);


  if (num_faces > mesh.nFaces()) {
    num_faces = mesh.nFaces(); // could probably optimize in this case by
                          // adding everything to the vector immediately
  }
  std::vector<FaceIter> ret;
  ret.reserve(num_faces);

  while(ret.size() < num_faces) {
    FaceIter f = proc_q.front();
    proc_q.pop();

    ret.push_back(f);

    HalfedgeIter& h = f->halfedge();
    for (int i = 0; i < 3; i++) {
      FaceIter f2 = h->twin()->face();
      if (all_seen.find(elementAddress(f2)) == all_seen.end()) {
        proc_q.push(f2);
        all_seen.insert(elementAddress(f2));
      }
      h = h->next();
    }
  }

  return ret;
}

Vector3D WaterColor::face_centroid(FaceIter f) {
  HalfedgeIter& h = f->halfedge();
  Vector3D cent;
  for (int i = 0; i < 3; i++) {
    cent += h->vertex()->position;
    h = h->next();
  }
  return cent / 3.;
}

Vector3D WaterColor::face_to_edge(Vector3D f_cent, HalfedgeIter h) {
  Vector3D to_edge = h->vertex()->position - f_cent;
  Vector3D edge_dir = (h->vertex()->position - h->twin()->vertex()->position).unit();
  return to_edge - dot(to_edge, edge_dir) * edge_dir;
}

float WaterColor::calc_angle(Vector3D f_cent, Vector3D n0, FaceIter f2, HalfedgeIter h) {

  Vector3D n2 = f2->normal();

  Vector3D x_axis = (face_to_edge(f_cent, h)).unit();
  float n_dot = dot(n0, n2);
  if (n_dot == 1.) {
    return 0.;
  }
  //n_dot should never be -1; that would imply the triangles overlap, violating a mesh assumption
  // all other n_dot will have some component in the "not n0 direction"
  float n2_x = dot(n2, x_axis);
  return atan2(n_dot, -n2_x);
}

/*
returns a number in the range [0, 1] representing the height of a face.
this is relatively computationally expensive, so if your simulation is slow
you might want to cache the results either inside or outside this function
(I'd recommend inside for a cleaner interface).

We could also look into less expensive estimates for face height than
intersection angle, or even just faster ways to implement the above algorithms
(maybe adding const and & in a few places is best practice?)
*/
float WaterColor::face_height(FaceIter f) {
  Vector3D f_cent = face_centroid(f);
  Vector3D n0 = f->normal();

  HalfedgeIter& h = f->halfedge();
  float tot_height = 0;
  for (int i = 0; i < 3; i++) {
    FaceIter f2 = h->twin()->face();
    tot_height += -1. * calc_angle(f_cent, n0, f2, h) / PI;
    h = h->next();
  }
  return (tot_height + 3.)/6.;
}

// main function for moving water in the shallow water layer
void WaterColor::move_water(std::vector<FaceIter> patch) {
  update_velocities(patch);
  // relax_divergence(patch);
  // flow_outward(patch);
}

// updating velocities, changing velocity in each direction by random factor scaled by height differential in that direction
void WaterColor::update_velocities(std::vector<FaceIter> patch) {
  //trying random velocity updates
  int timesteps = 1; // need to set timesteps if not following code from paper

  for (int t = 0; t < timesteps; t++) {

    for (FaceIter f: patch) {
      float x_rand_scale = abs(random_uniform());
      float y_rand_scale = abs(random_uniform());
      float z_rand_scale = abs(random_uniform());

      if (f->xyz_flow[0] > 0) {
        f->xyz_flow[0] = max(0.0, f->xyz_flow[0] - x_rand_scale * f->h_slopes[0]);
      } else if (f->xyz_flow[0] < 0) {
        f->xyz_flow[0] = min(0.0, f->xyz_flow[0] - x_rand_scale * f->h_slopes[0]);
      }

      if (f->xyz_flow[1] > 0) {
        f->xyz_flow[1] = max(0.0, f->xyz_flow[1] - y_rand_scale * f->h_slopes[1]);
      } else if (f->xyz_flow[0] < 0) {
        f->xyz_flow[1] = min(0.0, f->xyz_flow[1] - y_rand_scale * f->h_slopes[1]);
      }

      if (f->xyz_flow[2] > 0) {
        f->xyz_flow[2] = max(0.0, f->xyz_flow[2] - z_rand_scale * f->h_slopes[2]);
      } else if (f->xyz_flow[0] < 0) {
        f->xyz_flow[2] = min(0.0, f->xyz_flow[2] - z_rand_scale * f->h_slopes[2]);
      }

    }

    for (FaceIter f: patch) {
      if (!f->is_wc) {
        f->xyz_flow = Vector3D(0.0, 0.0, 0.0);
      }
    }
  }
}

//relaxing the divergence of the velocity field ???
void WaterColor::relax_divergence(std::vector<FaceIter> patch) {
  return;
}

//edge darkening, paper applies a gausian blur on water mask (faces with is_wc)
//can try another brute force way of only dealing with edge faces with is_wc
void WaterColor::flow_outward(std::vector<FaceIter> patch) {
  float remove_factor = 0.9; //tweak parameters??

  for (FaceIter f: patch) {

    HalfedgeIter& h = f->halfedge();
    for (int i = 0; i < 3; i++) {
      FaceIter f2 = h->twin()->face();

      if (!f2->is_wc) {
        f->pressure = f->pressure * remove_factor;
        break;
      }

      h = h->next();
    }
  }
}

//distribution of pigments from cell (face) to neighbors according to the rate of fluid movement out of the cell
//velocity in different directions result in different amounts of pigment movement => impacted by height differential
void WaterColor::move_pigment(std::vector<FaceIter> patch) {
  // float max_xyz = 0;
  // for (FaceIter f: patch) {
  //   max_xyz = max(max_xyz, ceil(max(max(abs(f->xyz_flow[0]), abs(f->xyz_flow[1])), abs(f->xyz_flow[2]))));
  // }

  int timesteps = 10; // need to set timesteps if not following code from paper

  for (int t = 0; t < timesteps; t++) {
    for (FaceIter f: patch) {
      f->pigments_g_new = f->pigments_g;
    }
    for (FaceIter f: patch) {

      std::vector<FaceIter> neighbors;
      HalfedgeIter& h = f->halfedge();
      for (int i = 0; i < 3; i++) {
        FaceIter f2 = h->twin()->face();
        neighbors.push_back(f2);
        h = h->next();
      }

      for (int i = 0; i < f->pigments_g.size(); i++) {
        float change = 0.0;
        if (neighbors[0]->is_wc) {
          neighbors[0]->pigments_g_new[i] += max(0.0, 1.5 * f->xyz_flow[0] * f->pigments_g[i]);
          change += max(0.0, f->xyz_flow[0] * f->pigments_g[i]);
        }
        if (neighbors[1]->is_wc) {
          neighbors[1]->pigments_g_new[i] += max(0.0, 1.5 * f->xyz_flow[1] * f->pigments_g[i]);
          change += max(0.0, f->xyz_flow[1] * f->pigments_g[i]);
        }
        if (neighbors[2]->is_wc) {
          neighbors[2]->pigments_g_new[i] += max(0.0, 1.5 * f->xyz_flow[2] * f->pigments_g[i]);
          change += max(0.0, f->xyz_flow[2] * f->pigments_g[i]);
        }
        f->pigments_g_new[i] -= change;
      }
    }
    for (FaceIter f: patch) {
      f->pigments_g = f->pigments_g_new;
    }
  }

}

//at each step of simulation, pigment is absorbed into the paper at certain rates, but also back into fluid at other rates
void WaterColor::transfer_pigment(std::vector<FaceIter> patch) {
  for (FaceIter f: patch) {
    if (f->is_wc) {
      for (int i = 0; i < f->pigments_g.size(); i++) {
        float delta_down = f->pigments_g[i] * (1.0 - f->height * f->granulation[i]) * f->density[i];
        float delta_up = f->pigments_d[i] * (1.0 + (f->height - 1.0) * f->granulation[i]) * f->density[i] / f->staining_power[i];
        if (f->pigments_d[i] + delta_down > 1) {
          delta_down = max(0.0, 1.0 - f->pigments_d[i]);
        }
        if (f->pigments_g[i] + delta_up > 1) {
          delta_up = max(0.0, 1.0 - f->pigments_g[i]);
        }
        f->pigments_d[i] += delta_down - delta_up;
        f->pigments_g[i] += delta_up - delta_down;
      }
    }
  }
}

//simulates how paper absorbs water and increases the water color area
void WaterColor::simulateCapillaryFlow(std::vector<FaceIter> patch) {
  float absorbtion_rate = 1.0; //tweak this parameter?
  float threshold = 0.5; //tweak this parameter?
  float min_saturation_diffuse_neighbors = 1.0; //tweak this parameter?
  float min_saturation_no_diffuse = 1.0; //tweak this parameter?

  for (FaceIter f: patch) {
    if (f->is_wc) {
      f->saturation += max(float(0.0), min(absorbtion_rate, f->capacity - f->saturation));
    }
    f->saturation_new = f->saturation;
  }

  for (FaceIter f: patch) {
    HalfedgeIter& h = f->halfedge();
    for (int i = 0; i < 3; i++) {
      FaceIter f2 = h->twin()->face();

      if (f->saturation > min_saturation_diffuse_neighbors && f->saturation > f2->saturation && f2->saturation > min_saturation_no_diffuse) {
        float delta_s = max(0.0, min(f->saturation - f2->saturation, f2->capacity - f2->saturation)/4.0);
        f->saturation_new -= delta_s;
        f2->saturation_new += delta_s;
      }

      h = h->next();
    }
  }

  for (FaceIter f: patch) {
    f->saturation = f->saturation_new;
    if (f->saturation > threshold) {
      f->is_wc = true;
    }
  }
}


// Simulates watercolor paint spreading over a mesh.
// NOTE: we eventually will need to have a way of specifying how
//       the brush-strokes are specified. For now we'll just initialize
//       stuff however we want, i.e. uniformly at random, or maybe using
//       the z coordinate(?) of the mesh element for testing.
void WaterColor::simulate_mesh(GLScene::Mesh* elem) {
  HalfedgeMesh& mesh = elem->get_underlying_mesh();

  /* List of all Face properties useful for simulation: (see util/halfEdgeMesh.h for definitions)
  Vector3D reflectance;
  Vector3D transmittance;
  */
  //for( FaceIter f = mesh.facesBegin(); f != mesh.facesEnd(); f++ ) //iterate over all faces

  std::vector<std::vector<FaceIter>> patches;

  StatsBuilder sb;
  for (int patch = 0; patch<5; patch++) {

    std::vector<FaceIter> newPatch = get_patch(mesh, 1500, true);
    patches.push_back(newPatch);

    
    //need to instantiate properties of each face "cell" for simulation
    for (FaceIter f: newPatch) {
      f->is_wc = true; // tells the renderer to use this face's watercolor reflectance instead of the default

      // initialize simulation params (below are placeholders; remove oor modify however you want)

      //f->wetness = (float)(random_uniform());
      float f_h = face_height(f);
      f->height = f_h;
      sb.add(f_h);

      //placeholder values, need to set default values
      f->xyz_flow = Vector3D(0.0, 0.0, 0.0); //need to set initial xyz_flow velocities to actually run update_velocities()?
      f->pressure = 1.0; //not used until we change fluid sim

      f->saturation = 1.0;
      f->saturation_new = 1.0;
      f->capacity = 1.0; //per paper, shoud be  c = h * (c_max - c_min) + c_min

      //all faces should have the same k pigments, varrying in property values
      //using placeholders, need to modify, can follow values in paper in figure 5
      f->pigments_g = {1.0, 0.1, 1.0};
      f->pigments_g_new = {1.0, 0.1, 1.0};
      f->density = {0.02, 0.09, 0.01};
      f->staining_power = {1.0, 1.0, 1.0};
      f->granulation = {0.63, 0.41, 0.31};
      f->pigments_d = {0.0, 0.0, 0.0};
    }
    sb.print_stats();
    sb.clear();

    //need to get a gradient of height differences for each direction, 3 in total
    for (FaceIter f: newPatch) {
      HalfedgeIter& h = f->halfedge();
      for (int i = 0; i < 3; i++) {
        FaceIter f2 = h->twin()->face();
        if (f2->is_wc) {
          f->h_slopes[i] = f->height - f2->height;
        } else {
          f->h_slopes[i] = 0.0;
        }
        h = h->next();
      }

      f->xyz_flow = f->h_slopes;
    }

  }

  // loop through all the patches
  // run water simulation on each patch
  // end result will result in each face having multiple properties of pigments
  // find a way to set reflectance and transmittance values of the face to show true water color pigments
  for (std::vector<FaceIter> patch : patches) {
    int timesteps = 15;
    for (int i = 0; i < timesteps; i++) {
      // for(FaceIter f : patch) {
      //   std::cout << f->xyz_flow[0] << "\t" << f->xyz_flow[1] << "\t" << f->xyz_flow[2] << endl;
      // }
      // for(FaceIter f : patch) {
      //   std::cout << f->pigments_g[0] << "\t" << f->pigments_g[1] << "\t" << f->pigments_g[2] << endl;
      // }
      move_water(patch);
      // std::cout << "DONE WITH MOVING WATER" << endl;
      move_pigment(patch);
      // for(FaceIter f : patch) {
      //   std::cout << f->pigments_g[0] << "\t" << f->pigments_g[1] << "\t" << f->pigments_g[2] << endl;
      // }
      // std::cout << "DONE WITH MOVING PIGMENTS" << endl;
      transfer_pigment(patch);
      // simulateCapillaryFlow(patch);
      // for(FaceIter f : patch) {
      //   std::cout << f->xyz_flow[0] << "\t" << f->xyz_flow[1] << "\t" << f->xyz_flow[2] << endl;
      // }
      // for(FaceIter f : patch) {
      //   std::cout << f->pigments_g[0] << "\t" << f->pigments_g[1] << "\t" << f->pigments_g[2] << endl;
      // }
      // std::cout << "PIGMENT D" << endl;
      // for(FaceIter f : patch) {
      //   std::cout << f->pigments_d[0] << "\t" << f->pigments_d[1] << "\t" << f->pigments_d[2] << endl;
      // }
    }
  }
  for (std::vector<FaceIter> patch : patches) {
    // for(FaceIter f : patch) {
    //   //example render of heightmap
    //   if (f->height <0.2){
    //     f->reflectance = Vector3D(0.6f, 0.1f, 0.1f);
    //   } else if (f->height < 0.25) {
    //     f->reflectance = Vector3D(0.1f, 0.6f, 0.1f);
    //   } else {
    //     f->reflectance = Vector3D(0.1f, 0.1f, 0.6f);
    //   }
    // }



    for(FaceIter f : patch) {
      float pigment_d_sum = 0.0;
      std::vector<float> factors;
      for (int i = 0; i < f->pigments_d.size(); i++) {
        pigment_d_sum += f->pigments_g[i] + f->pigments_d[i];
      }
      for (int i = 0; i < f->pigments_d.size(); i++) {
        if (pigment_d_sum == 0) {
          factors.push_back(0.0);
        } else {
          factors.push_back((f->pigments_g[i]+f->pigments_d[i])/1.2);
          // std::cout << (f->pigments_g[i] + f->pigments_d[i])/pigment_d_sum << endl;
        }
      }

      Vector3D custom_r = Vector3D(207.0/255.0, 82.0/255.0,  75.0/255.0);
      Vector3D custom_g = Vector3D(87.0/255.0, 162.0/255.0,  75.0/255.0);
      Vector3D custom_b = Vector3D(64.0/255.0, 125.0/255.0,  237.0/255.0);
      //std::cout << factors[0] << "\t" << factors[1] << "\t" << factors[2] << endl;
      // std::cout << factors[0] + factors[1] + factors[2] << endl;
      f->reflectance = factors[0] * custom_r + factors[1] * custom_g + factors[2] * custom_b;
      // f->transmittance = factors[0] * custom_r + factors[1] * custom_g + factors[2] * custom_b;
      // f->reflectance = (1.0/2.1) * custom_r + (0.1/2.1) * custom_g + (1.0/2.1) * custom_b;
      // f->reflectance = Vector3D(factors[0],factors[1],factors[2]);
    }
    // Vector3D red = Vector3D(0.77,  0.015,  0.018);
    // Vector3D green = Vector3D(0.01,  0.012,  0.003);

  }


}

} // namespace CGL
