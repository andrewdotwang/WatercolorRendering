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

}  // namespace CGL

#endif  // CGL_RAYTRACER_H
