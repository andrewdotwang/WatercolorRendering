#ifndef CGL_WATERCOLOR_H
#define CGL_WATERCOLOR_H

#include "CGL/timer.h"

#include "scene/bvh.h"
#include "pathtracer/sampler.h"
#include "pathtracer/intersection.h"

#include "application/renderer.h"

#include "scene/gl_scene/mesh.h"
#include "scene/scene.h"
using CGL::SceneObjects::Scene;

#include "scene/environment_light.h"
using CGL::SceneObjects::EnvironmentLight;

using CGL::SceneObjects::BVHNode;
using CGL::SceneObjects::BVHAccel;

#include <unordered_set>
#include <queue>

namespace CGL {

    class WaterColor {
    public:
        WaterColor();
        ~WaterColor();

        void simulate(GLScene::SceneObject *elem);

    private:
    	void simulate_mesh(GLScene::Mesh* elem);
    	std::vector<FaceIter> get_patch(HalfedgeMesh& mesh, int num_faces, bool visible);
    };

    // I think we actually only need reflectance for rendering, probably could
    // remove transmittance unless we're doing translucent wet paper
	struct WcInfo {

	  // Default constructor.
	  WcInfo() : WcInfo(Vector3D(), Vector3D(), false) { }

	  WcInfo(Vector3D r, Vector3D t)
	      : reflectance(r), transmittance(t), is_wc(true) {}

	  WcInfo(Vector3D r, Vector3D t, bool b)
	      : reflectance(r), transmittance(t), is_wc(b) {}	      

	  Vector3D reflectance;
	  Vector3D transmittance;
	  bool is_wc;
	};

}  // namespace CGL

#endif  // CGL_RAYTRACER_H
