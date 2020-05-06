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

void WaterColor::move_water(std::vector<FaceIter> patch) {
  return;
}

void WaterColor::update_velocities(std::vector<FaceIter> patch) {
  //trying random velocity updates
  for (FaceIter f: patch) {
    float x_rand_scale = abs(random_uniform());
    float y_rand_scale = abs(random_uniform());
    float z_rand_scale = abs(random_uniform());

    if (f->xyz_flow[0] > 0) {
      f->xyz_flow[0] = max(0.0, f->xyz_flow[0] - x_rand_scale * f->h_slopes[0]);
    } else if (f->xyz_flow[0] < 0) {
      f->xyz_flow[0] = min(0.0, f->xyz_flow[0] + x_rand_scale * f->h_slopes[0]);
    }

    if (f->xyz_flow[1] > 0) {
      f->xyz_flow[1] = max(0.0, f->xyz_flow[1] - y_rand_scale * f->h_slopes[1]);
    } else if (f->xyz_flow[0] < 0) {
      f->xyz_flow[1] = min(0.0, f->xyz_flow[1] + y_rand_scale * f->h_slopes[1]);
    }

    if (f->xyz_flow[2] > 0) {
      f->xyz_flow[2] = max(0.0, f->xyz_flow[2] - z_rand_scale * f->h_slopes[2]);
    } else if (f->xyz_flow[0] < 0) {
      f->xyz_flow[2] = min(0.0, f->xyz_flow[2] + z_rand_scale * f->h_slopes[2]);
    }

  }

  // int max_uv = 0;
  // for (FaceIter f: newPatch) {
  //   f->xyz_flow = f->xyz_flow - f->h_slope;
    // max_uv = max(max_uv, ceil(max(f->xyz_flow[0],f->xyz_flow[1])));
  // }
  //
  // for (int t = 0; t < 1; t += 1/max_uv) {
  //   for (FaceIter f: newPatch) {
  //   }
  // }

}

void WaterColor::flow_outward(std::vector<FaceIter> patch) {
  //makes edge darken
  return;
}

void WaterColor::move_pigment(std::vector<FaceIter> patch) {
  // float max_xyz = 0;
  // for (FaceIter f: patch) {
  //   max_xyz = max(max_xyz, ceil(max(max(abs(f->xyz_flow[0]), abs(f->xyz_flow[1])), abs(f->xyz_flow[2]))));
  // }

  for (int t = 0; t < 5; t++) {

    for (FaceIter f: patch) {

      std::vector<FaceIter> neighbors;
      HalfedgeIter& h = f->halfedge();
      for (int i = 0; i < 3; i++) {
        FaceIter f2 = h->twin()->face();
        neighbors.push_back(f2);
        h = h->next();
      }

      for (int i = 0; i < f->pigments_g.size(); i++) {
        neighbors[0]->pigments_g_new[i] = neighbors[0]->pigments_g_new[i] + max(0.0, f->xyz_flow[0] * f->pigments_g[i]);
        neighbors[1]->pigments_g_new[i] = neighbors[1]->pigments_g_new[i] + max(0.0, f->xyz_flow[1] * f->pigments_g[i]);
        neighbors[2]->pigments_g_new[i] = neighbors[2]->pigments_g_new[i] + max(0.0, f->xyz_flow[2] * f->pigments_g[i]);
        f->pigments_g_new[i] = f->pigments_g_new[i] - max(0.0, f->xyz_flow[0] * f->pigments_g[i]) - max(0.0, f->xyz_flow[1] * f->pigments_g[i]) - max(0.0, f->xyz_flow[2] * f->pigments_g[i]);
      }
    }
  }

}

void WaterColor::transfer_pigment(std::vector<FaceIter> patch) {
  for (FaceIter f: patch) {
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

// void WaterColor::simulateCapillaryFlow(std::vector<FaceIter> patch) {
//
// }


// Simulates watercolor paint spreading over a mesh.
// NOTE: we eventually will need to have a way of specifying how
//       the brush-strokes are specified. For now we'll just initialize
//       stuff however we want, i.e. uniformly at random, or maybe using
//       the z coordinate(?) of the mesh element for testing.
void WaterColor::simulate_mesh(GLScene::Mesh* elem) {
  HalfedgeMesh& mesh = elem->get_underlying_mesh();

  // I added a bunch of properties to each mesh face that I thought would be useful
  // you can access and set them all using i.e. f->xyz_flow = Vector2D(0.2, 0.3);
  /* List of all Face properties useful for simulation: (see util/halfEdgeMesh.h for definitions)
  float wetness;
  Vector3D xyz_flow;
  float pressure;

  std::vector<float> pigments_g;
  std::vector<float> pigments_d;

  bool is_wc;
  Vector3D reflectance;
  Vector3D transmittance;
  */
  //for( FaceIter f = mesh.facesBegin(); f != mesh.facesEnd(); f++ ) //iterate over all faces
  for (int patch = 0; patch<5; patch++) {
    std::vector<FaceIter> newPatch = get_patch(mesh, 400, true);
    for (FaceIter f: newPatch) {
      f->is_wc = true; // tells the renderer to use this face's watercolor reflectance instead of the default

      // initialize simulation params (below are placeholders; remove oor modify however you want)

      //f->wetness = (float)(random_uniform());
      float f_h = face_height(f);
      f->height = f_h;
      f->xyz_flow = Vector3D(0.0, 0.0, 0.0);

      if (f_h <0.2){
        f->reflectance = Vector3D(0.6f, 0.1f, 0.1f);
      } else if (f_h < 0.25) {
        f->reflectance = Vector3D(0.1f, 0.6f, 0.1f);
      } else {
        f->reflectance = Vector3D(0.1f, 0.1f, 0.6f);
      }
      //f->reflectance = Vector3D(f_h,f_h,f_h);
      //f->reflectance = Vector3D(0.6f, 0.1f, 0.1f);
    }

    for (FaceIter f: newPatch) {
      HalfedgeIter& h = f->halfedge();
      for (int i = 0; i < 3; i++) {
        FaceIter f2 = h->twin()->face();
        f->h_slopes[i] = f->height - f2->height;
        h = h->next();
      }
    }

  }

  // loop through all the patches
  // run water simulation on each patch
  // end result will result in each face having multiple properties of pigments
  // find a way to set reflectance and transmittance values of the face to show true water color pigments
}

} // namespace CGL
