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

namespace CGL {

    class WaterColor {
    public:
        WaterColor();
        ~WaterColor();

        void simulate(GLScene::SceneObject *elem);

    private:
    	void simulate_mesh(GLScene::Mesh* elem);
    };

	struct WcInfo {

	  // Default constructor.
	  WcInfo() : WcInfo(Vector3D(), Vector3D()) { }

	  WcInfo(Vector3D r, Vector3D t)
	      : reflectance(r), transmittance(t) {}

	  Vector3D reflectance;
	  Vector3D transmittance;

	};

}  // namespace CGL

#endif  // CGL_RAYTRACER_H
