#include "bbox.h"

#include "GL/glew.h"

#include <algorithm>
#include <iostream>

namespace CGL {

bool BBox::intersect(const Ray& r, double& t0, double& t1) const {

  // TODO (Part 2.2):
  // Implement ray - bounding box intersection test
  // If the ray intersected the bouding box within the range given by
  // t0, t1, update t0 and t1 with the new intersection times.
  /*
  Vector3D cent_min = min - r.o;
  Vector3D cent_max = max - r.o;

  //look along x axis for intersections
  
  double t1x = cent_min[0]/r.d[0];
  double t2x = cent_max[0]/r.d[0];
  if (t1x > t2x) {
    std::swap(t1x, t2x);
  }

  //same for y 
  double t1y = cent_min[1]/r.d[1];
  double t2y = cent_max[1]/r.d[1];
  if (t1y > t2y) {
    std::swap(t1y, t2y);
  }

  //same for z 
  double t1z = cent_min[2]/r.d[2];
  double t2z = cent_max[2]/r.d[2];
  if (t1z > t2z) {
    std::swap(t1z, t2z);
  }

  double t_min = std::max(t1x, std::max(t1y, t1z));
  double t_max = std::min(t2x, std::min(t2y, t2z));
  */

  Vector3D cent_min = (min - r.o)* (1/r.d);
  Vector3D cent_max = (max - r.o) * (1/r.d);
  double t_min, t_max;
  t_min = cent_min[0];
  t_max = cent_max[0];
  if (t_min > t_max) {
    std::swap(t_min, t_max);
  }

  if (cent_min[1] < cent_max[1]) {
    t_min = std::max(t_min, cent_min[1]);
    t_max = std::min(t_max, cent_max[1]);
  } else {
    t_min = std::max(t_min, cent_max[1]);
    t_max = std::min(t_max, cent_min[1]);
  }

  if (cent_min[2] < cent_max[2]) {
    t_min = std::max(t_min, cent_min[2]);
    t_max = std::min(t_max, cent_max[2]);
  } else {
    t_min = std::max(t_min, cent_max[2]);
    t_max = std::min(t_max, cent_min[2]);
  }

  if ((t_min > t_max) || t_min > r.max_t || t_max < r.min_t) {
    return false;
  }

  t0 = t_min;
  t1 = t_max;
  return true;
}

void BBox::draw(Color c, float alpha) const {

  glColor4f(c.r, c.g, c.b, alpha);

  // top
  glBegin(GL_LINE_STRIP);
  glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, max.y, min.z);
  glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, max.y, max.z);
  glVertex3d(max.x, max.y, max.z);
  glEnd();

  // bottom
  glBegin(GL_LINE_STRIP);
  glVertex3d(min.x, min.y, min.z);
  glVertex3d(min.x, min.y, max.z);
  glVertex3d(max.x, min.y, max.z);
  glVertex3d(max.x, min.y, min.z);
  glVertex3d(min.x, min.y, min.z);
  glEnd();

  // side
  glBegin(GL_LINES);
  glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, min.y, max.z);
  glVertex3d(max.x, max.y, min.z);
  glVertex3d(max.x, min.y, min.z);
  glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, min.y, min.z);
  glVertex3d(min.x, max.y, max.z);
  glVertex3d(min.x, min.y, max.z);
  glEnd();

}

std::ostream& operator<<(std::ostream& os, const BBox& b) {
  return os << "BBOX(" << b.min << ", " << b.max << ")";
}

} // namespace CGL
