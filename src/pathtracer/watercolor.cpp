#include "watercolor.h"

#include "scene/light.h"
#include "scene/sphere.h"
#include "scene/triangle.h"


using namespace CGL::SceneObjects;

namespace CGL {

WaterColor::WaterColor() {
  gridSampler = new UniformGridSampler2D();
  hemisphereSampler = new UniformHemisphereSampler3D();

  tm_gamma = 2.2f;
  tm_level = 1.0f;
  tm_key = 0.18;
  tm_wht = 5.0f;
}

WaterColor::~WaterColor() {
  delete gridSampler;
  delete hemisphereSampler;
}

void WaterColor::set_frame_size(size_t width, size_t height) {
  sampleBuffer.resize(width, height);
  sampleCountBuffer.resize(width * height);
}

void WaterColor::clear() {
  bvh = NULL;
  scene = NULL;
  camera = NULL;
  sampleBuffer.clear();
  sampleCountBuffer.clear();
  sampleBuffer.resize(0, 0);
  sampleCountBuffer.resize(0, 0);
}

void WaterColor::simulate() {
	
}

void WaterColor::write_to_framebuffer(ImageBuffer &framebuffer, size_t x0,
                                      size_t y0, size_t x1, size_t y1) {
  sampleBuffer.toColor(framebuffer, x0, y0, x1, y1);
}

Spectrum
WaterColor::estimate_direct_lighting_hemisphere(const Ray &r,
                                                const Intersection &isect) {
  // Estimate the lighting from this intersection coming directly from a light.
  // For this function, sample uniformly in a hemisphere.

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  const Vector3D &hit_p = r.o + r.d * isect.t;
  const Vector3D &w_out = w2o * (-r.d);

  // This is the same number of total samples as
  // estimate_direct_lighting_importance (outside of delta lights). We keep the
  // same number of samples for clarity of comparison.
  int num_samples = scene->lights.size() * ns_area_light;
  Spectrum L_out;

  // TODO (Part 3): Write your sampling loop here
  // TODO BEFORE YOU BEGIN
  // UPDATE `est_radiance_global_illumination` to return direct lighting instead of normal shading 
  Intersection isect2;
  Spectrum em, refl;
  double cos_term;
  Vector3D samp, w_in;
  Ray r2((Vector3D()), (Vector3D()));

  for (int i = 0; i < num_samples; i++) {
    samp = hemisphereSampler->get_sample();
    w_in = o2w * samp; // note: world frame!
    r2 = Ray((hit_p + EPS_D * w_in), w_in);
    if (bvh->intersect(r2, &isect2)) {
      //emission intensity from the sampled ray
      em = isect2.bsdf->get_emission();

      if ((em[0] > 0) || (em[1] > 0) || (em[2] > 0)) {
        // bsdf of surface
        refl = isect.bsdf->f(w_out, samp);

        //cosine of the incoming ray
        cos_term = cos_theta(samp.unit());

        //reflection equation monte carlo (pdf is 1/(2pi))
        L_out += (refl * em * cos_term * 2 * PI);
      }
    }
  }

  return L_out / float(num_samples);
}

Spectrum
WaterColor::estimate_direct_lighting_importance(const Ray &r,
                                                const Intersection &isect) {
  // Estimate the lighting from this intersection coming directly from a light.
  // To implement importance sampling, sample only from lights, not uniformly in
  // a hemisphere.

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  const Vector3D &hit_p = r.o + r.d * isect.t;
  const Vector3D &w_out = w2o * (-r.d);
  Spectrum L_out;

  Intersection isect2;
  Spectrum em2, refl;
  double cos_term;
  Vector3D w_in;
  //Ray r2((Vector3D()), (Vector3D()));
  int numsamp = ns_area_light;
  int tot_samps = 0;
  float dist_to_light, pdf;

  for (auto l = scene->lights.begin(); l != scene->lights.end(); l++) {
    numsamp = ns_area_light;
    if ((*l)->is_delta_light()) {
      numsamp = 1;
    }
    
    for (int i = 0; i < numsamp; i++) {
      Spectrum em = ((*l)->sample_L(hit_p, &w_in, &dist_to_light, &pdf));
      Vector3D samp = w2o * w_in; // note: object frame

      Ray r2 = Ray(hit_p, w_in, dist_to_light - EPS_F);
      r2.min_t = EPS_F;

      tot_samps += 1;
      //(dot(w_in, isect.n) > 0) &&
      if ((dot(w_in, isect.n) > 0) && !bvh->has_intersection(r2)) {
        //emission intensity from the sampled ray

        //em2 = isect2.bsdf->get_emission();

        //if (fabs(isect2.t - dist_to_light) < EPS_F) {
          // bsdf of surface
          refl = isect.bsdf->f(w_out, samp);

          //cosine of the incoming ray
          cos_term = cos_theta(samp.unit());

          //reflection equation monte carlo (pdf is 1/(2pi))
          L_out += refl * em * cos_term / (float(pdf) * numsamp);
          
        //}
      }
    }


  }

  //cout << L_out <<endl;

  return L_out;
}

Spectrum WaterColor::zero_bounce_radiance(const Ray &r,
                                          const Intersection &isect) {
  // TODO: Part 3, Task 2
  // Returns the light that results from no bounces of light

  return isect.bsdf->get_emission();
  //return Spectrum(1.0);
}

Spectrum WaterColor::one_bounce_radiance(const Ray &r,
                                         const Intersection &isect) {
  // TODO: Part 3, Task 3
  // Returns either the direct illumination by hemisphere or importance sampling
  // depending on `direct_hemisphere_sample`

  if (direct_hemisphere_sample) {
    return estimate_direct_lighting_hemisphere(r, isect);
  }
  return estimate_direct_lighting_importance(r, isect);
}

Spectrum WaterColor::at_least_one_bounce_radiance(const Ray &r,
                                                  const Intersection &isect) {
  Spectrum L_out(0.,0.,0.);
  //if (r.depth < max_ray_depth) {  // for the "only indirect lighting" experiment
    L_out = one_bounce_radiance(r, isect);
  //}
  
  //

  if (r.depth <= 1 || !coin_flip(russian_prob)) {
    return L_out;
    //return one_bounce_radiance(r, isect);
  }

  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  Vector3D hit_p = r.o + r.d * isect.t;
  Vector3D w_out = w2o * (-r.d);


  Vector3D samp, w_in;
  float pdf;
  Spectrum em = isect.bsdf->sample_f(w_out, &samp, &pdf);

  w_in = o2w * samp;
  Ray r2(hit_p + EPS_D * w_in, w_in, INF_D, r.depth - 1);

  Intersection isect2;
  if (bvh->intersect(r2, &isect2)) {
    //isect2.bsdf->f(w_out, samp)
    L_out += at_least_one_bounce_radiance(r2, isect2) * em * cos_theta(samp.unit()) / (pdf * russian_prob);
  }

  return L_out;
}

Spectrum WaterColor::est_radiance_global_illumination(const Ray &r) {
  Intersection isect;
  Spectrum L_out;

  // You will extend this in assignment 3-2.
  // If no intersection occurs, we simply return black.
  // This changes if you implement hemispherical lighting for extra credit.

  if (!bvh->intersect(r, &isect))
    return L_out;

  // The following line of code returns a debug color depending
  // on whether ray intersection with triangles or spheres has
  // been implemented.

  // REMOVE THIS LINE when you are ready to begin Part 3.
  //L_out = (isect.t == INF_D) ? debug_shading(r.d) : normal_shading(isect.n);

  // TODO (Part 3): Return the direct illumination.

  //return one_bounce_radiance(r, isect); //only direct illumination
  L_out = zero_bounce_radiance(r, isect);

  if (max_ray_depth > 0) {
    if (PART <= 3 || max_ray_depth == 1) {
      L_out += one_bounce_radiance(r, isect);
    } else {
      L_out += at_least_one_bounce_radiance(r, isect);
    }
  }




  // TODO (Part 4): Accumulate the "direct" and "indirect"
  // parts of global illumination into L_out rather than just direct

  return L_out;
}

void WaterColor::raytrace_pixel(size_t x, size_t y) {

  // TODO (Part 1.1):
  // Make a loop that generates num_samples camera rays and traces them
  // through the scene. Return the average Spectrum.
  // You should call est_radiance_global_illumination in this function.

  // TODO (Part 5):
  // Modify your implementation to include adaptive sampling.
  // Use the command line parameters "samplesPerBatch" and "maxTolerance"

  int num_samples = ns_aa;          // total samples to evaluate
  Vector2D origin = Vector2D(x, y); // bottom left corner of the pixel

  Spectrum ret, temp;
  Vector2D samp;
  Ray r((Vector3D()), (Vector3D()), INF_D, max_ray_depth); // Why is C++ so bad...

  float s1 = 0;
  float s2 = 0;
  float mu = 0;
  float sig2 = 0;
  float ill = 0;
  int i = 0;

  for (; i < num_samples; i++) {
    samp = gridSampler->get_sample();
    //cout << samp << endl;
    float norm_x = (samp[0] + origin[0])/float(sampleBuffer.w);
    float norm_y = (samp[1] + origin[1])/float(sampleBuffer.h);
    r = camera->generate_ray(norm_x, norm_y);
    r.depth = max_ray_depth;
    temp = est_radiance_global_illumination(r);
    ret += temp;
    if (PART == 5) {
      ill = temp.illum();
      s1 += ill;
      s2 += ill * ill;
      if ((i+1) % samplesPerBatch == 0) {
        mu = s1 / float(i + 1);
        sig2 = (s2 - s1*mu)/ float(i);
        if (1.96 * sqrt(sig2/float(i+1)) <= maxTolerance * mu) {
          break;
        }
      }
    }

  }
  ret = ret/(i+1);
  
  sampleBuffer.update_pixel(ret, x, y);
  sampleCountBuffer[x + y * sampleBuffer.w] = i + 1;
}

} // namespace CGL