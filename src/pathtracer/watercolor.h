#ifndef CGL_WATERCOLOR_H
#define CGL_WATERCOLOR_H

#include "CGL/timer.h"

#include "scene/bvh.h"
#include "pathtracer/camera.h"
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
#include <random>
#include <time.h>

namespace CGL {

    class WaterColor {
    public:
        WaterColor();
        ~WaterColor();

        void simulate(GLScene::SceneObject *elem);

        Camera* camera;       ///< current camera
    private:
    	bool can_see(FaceIter f);
    	FaceIter get_random_face(HalfedgeMesh& mesh);
    	void simulate_mesh(GLScene::Mesh* elem);
    	std::vector<FaceIter> get_patch(HalfedgeMesh& mesh, int num_faces, bool visible);

    	Vector3D face_centroid(FaceIter f);
		Vector3D face_to_edge(Vector3D f_cent, HalfedgeIter h);
    	float calc_angle(Vector3D f_cent, Vector3D n0, FaceIter f2, HalfedgeIter h);
    	float face_height(FaceIter f);
      void move_water(std::vector<FaceIter> patch);
      void update_velocities(std::vector<FaceIter> patch);
      void relax_divergence(std::vector<FaceIter> patch);
      void flow_outward(std::vector<FaceIter> patch);
      void move_pigment(std::vector<FaceIter> patch);
      void transfer_pigment(std::vector<FaceIter> patch);
      void simulateCapillaryFlow(std::vector<FaceIter> patch);


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

	struct Stats {

		Stats() : Stats(0., 0., 0., 0., 0) { }

		Stats(float me, float st, float mi, float ma, int nv)
		      : mean(me), std(st), min(mi), max(ma), num_vals(nv) {}

		float mean;
		float std;
		float min;
		float max;
		float num_vals;
	};

	class StatsBuilder {
		public:
	        StatsBuilder();

	        void add(float n);
	        Stats calc_stats();
	        void print_stats();
	        void clear();
          float get_mean();
          float get_std();
	    private:
	    	std::vector<float> nums;
	    	float sum;
	};



}  // namespace CGL

#endif  // CGL_RAYTRACER_H
