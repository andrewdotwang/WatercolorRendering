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

/* a nice method used to get a patch of faces of a given size on the mesh. 

\param mesh The HalfedgeMesh from which we will return a set of touching faces
\param num_faces The number of faces that should be in the returned patch 
\param visible if True, this indicates that the patch should be somewhere visible
               to the camera.
*/
std::vector<FaceIter> WaterColor::get_patch(HalfedgeMesh& mesh, int num_faces, bool visible) {
  std::unordered_set<Face*> all_seen;

  //start with random face
  unsigned long start_ind = (unsigned long)(random_uniform() * mesh.nFaces());
  FaceIter starter_f = mesh.facesBegin();
  for(int tmp = 0; tmp < start_ind; tmp++) {
    starter_f++;
  }
  //FaceIter starter_f = mesh.facesBegin() + ;

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

// Simulates watercolor paint spreading over a mesh.
// NOTE: we eventually will need to have a way of specifying how
//       the brush-strokes are specified. For now we'll just initialize
//       stuff however we want, i.e. uniformly at random, or maybe using
//       the z coordinate(?) of the mesh element for testing.
void WaterColor::simulate_mesh(GLScene::Mesh* elem) {
  HalfedgeMesh& mesh = elem->get_underlying_mesh();

  // I added a bunch of properties to each mesh face that I thought would be useful
  // you can access and set them all using i.e. f->uv_flow = Vector2D(0.2, 0.3);
  /* List of all Face properties useful for simulation: (see util/halfEdgeMesh.h for definitions)
  float wetness;
  Vector2D uv_flow;
  float pressure;

  std::vector<float> pigments_g;
  std::vector<float> pigments_d;

  bool is_wc;
  Vector3D reflectance;
  Vector3D transmittance;
  */
  //for( FaceIter f = mesh.facesBegin(); f != mesh.facesEnd(); f++ ) //iterate over all faces
  for (FaceIter f: get_patch(mesh, 200, false))
  {
    f->is_wc = true; // tells the renderer to use this face's watercolor reflectance instead of the default

    // initialize simulation params (below are placeholders; remove oor modify however you want)
    f->wetness = (float)(random_uniform());
    f->reflectance = Vector3D(0.6f, 0.1f, 0.1f);
  }
}

} // namespace CGL