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
    //cout << "ssuh" << endl;
	  if (dynamic_cast<GLScene::Mesh*>(elem)) { 
      //cout << "hulloooooooooooooooooooooooooooooooooooooooooooo" << endl;
      simulate_mesh((GLScene::Mesh*)elem);
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
  float wetness;
  Vector2D uv_flow;
  float pressure;

  std::vector<float> pigments_g;
  std::vector<float> pigments_d;

  Vector3D reflectance;
  Vector3D transmittance;
  */
  for( FaceIter f = mesh.facesBegin(); f != mesh.facesEnd(); f++ )
  {
    f->wetness = (float)(random_uniform());
    f->reflectance = Vector3D(0.6f, 0.1f, 0.1f);
    // do something interesting with v
  }
}

} // namespace CGL