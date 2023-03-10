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
#include <math.h>

namespace CGL {

	class WC_Color {
		public:
	        WC_Color(Vector3D K2, Vector3D S2, float dens, float st, float gran);
	        WC_Color(Vector3D raw_color);

	        void calc_optics(float thickness, Vector3D *trans, Vector3D *refl);
	        //void calc_comp(Color& c2, Vector3D *trans, Vector3D *refl);
	    
	    	Vector3D K;
	    	Vector3D S;
	    	float density;
	    	float stain;
	    	float granulation;
		private:
	    	Vector3D a;
	    	Vector3D b;
	};

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

	  void get_timescale(std::vector<FaceIter>& patch, float* timesteps, float* dt);
	  void calc_vel_coords(float* x_coord, float* y_coord, Vector3D& x_axis, Vector3D& y_axis, HalfedgeIter& h);
	  float scalar_proj(Vector3D& v, Vector3D& axis);
	  Vector3D proj(Vector3D& v, Vector3D& axis);

      
      float mu = 0.1;
      float kappa = 0.01;

      WC_Color calc_comp(std::vector<WC_Color>& colors, std::vector<float> thicknesses, float *tot_thickness);
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
