#include "bvh.h"

#include "CGL/CGL.h"
#include "triangle.h"

#include <iostream>
#include <stack>

using namespace std;

namespace CGL {
namespace SceneObjects {

BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {

  primitives = std::vector<Primitive *>(_primitives);
  root = construct_bvh(primitives.begin(), primitives.end(), max_leaf_size);
}

BVHAccel::~BVHAccel() {
  if (root)
    delete root;
  primitives.clear();
}

BBox BVHAccel::get_bbox() const { return root->bb; }

void BVHAccel::draw(BVHNode *node, const Color &c, float alpha) const {
  if (node->isLeaf()) {
    for (auto p = node->start; p != node->end; p++) {
      (*p)->draw(c, alpha);
    }
  } else {
    draw(node->l, c, alpha);
    draw(node->r, c, alpha);
  }
}

void BVHAccel::drawOutline(BVHNode *node, const Color &c, float alpha) const {
  if (node->isLeaf()) {
    for (auto p = node->start; p != node->end; p++) {
      (*p)->drawOutline(c, alpha);
    }
  } else {
    drawOutline(node->l, c, alpha);
    drawOutline(node->r, c, alpha);
  }
}


BVHNode *BVHAccel::construct_bvh(std::vector<Primitive *>::iterator start,
                                 std::vector<Primitive *>::iterator end,
                                 size_t max_leaf_size) {

  // TODO (Part 2.1):
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.

  BBox bbox;

  int count = 0;
  for (auto p = start; p != end; p++) {
    BBox bb = (*p)->get_bbox();
    bbox.expand(bb);
    count += 1;
  }

  BVHNode *node = new BVHNode(bbox);

  if (PART == 1 || count <= max_leaf_size || start == end) {
    node->start = start;
    node->end = end;
    node->l = NULL;
    node->r = NULL;
    return node;
  }

  int axis = 0;
  Vector3D cent = bbox.extent;
  if (cent[1] > cent[0]) {
    axis = 1;
  } else if (cent[2] > cent[0]) {
    axis = 2;
  }

  std::function<bool(Primitive*, Primitive*)> comp = [&axis](Primitive *p1, Primitive *p2) {
    return p1->get_bbox().centroid()[axis] > p2->get_bbox().centroid()[axis];
  };

  // sort along axis
  std::sort(start, end, comp);

  // does this actually work? if so, spicy C++ magic
  int it = 0;
  std::vector<Primitive *>::iterator mid_end, mid_start;
  for (auto p = start; p != end; p++) {
    it += 1;
    if (it >= count / 2) {
      mid_end = p;
      mid_start = p;
      break;
    }
  }

  node->l = construct_bvh(start, mid_end, max_leaf_size);
  node->r = construct_bvh(mid_start, end, max_leaf_size);
  return node;
}

bool BVHAccel::has_intersection(const Ray &ray, BVHNode *node) const {
  // TODO (Part 2.3):
  // Fill in the intersect function.
  // Take note that this function has a short-circuit that the
  // Intersection version cannot, since it returns as soon as it finds
  // a hit, it doesn't actually have to find the closest hit.

  if (PART == 1) {
    for (auto p : primitives) {
      total_isects++;
      if (p->has_intersection(ray))
        return true;
    }
    return false;
  }

  std::stack<BVHNode*> nodesToProc;
  BVHNode *curr;
  nodesToProc.push(node);
  double t0, t1;
  while(!nodesToProc.empty()) {
    curr = nodesToProc.top();
    nodesToProc.pop();
    total_isects++;
    if (!curr->bb.intersect(ray, t0, t1)) {
      continue;
    }

    if (curr->l == NULL) {
      for (auto p = curr->start; p != curr->end; p++) {
        total_isects++;
        if ((*p)->has_intersection(ray)) {
          return true;
        }
      }
      continue;
    }

    nodesToProc.push(curr->l);
    nodesToProc.push(curr->r);
  }
  return false;

}


bool BVHAccel::intersect(const Ray &ray, Intersection *i, BVHNode *node) const {
  // TODO (Part 2.3):
  // Fill in the intersect function.

  if (PART == 1) {
    bool hit = false;
    for (auto p : primitives) {
      total_isects++;
      hit = p->intersect(ray, i) || hit;
    }
    return hit;
  }

  double t0, t1;
  std::stack<BVHNode*> nodesToProc;
  if (node->bb.intersect(ray, t0, t1)) {
    nodesToProc.push(node);
  }

  BVHNode *curr;
  bool hit = false;

  while(nodesToProc.size() > 0) {

    total_isects++;
    curr = nodesToProc.top();
    nodesToProc.pop();

    if (curr->isLeaf()) {

      for (auto p = curr->start; p != curr->end; p++) {
        total_isects++;
        hit = (*p)->intersect(ray, i) || hit;
      }
      
      continue;
    }
    if (curr->l->bb.intersect(ray, t0, t1)) {
      nodesToProc.push(curr->l);
    }
    if (curr->r->bb.intersect(ray, t0, t1)) {
      nodesToProc.push(curr->r);
    }
  }


  return hit;
}

} // namespace SceneObjects
} // namespace CGL
