#include "scene.h"
#include "mesh.h" // no idea why this has to go here and not in the header

using std::cout;
using std::endl;

namespace CGL { namespace GLScene {

BBox Scene::get_bbox() {
  BBox bbox;
  for (SceneObject *obj : objects) {
    bbox.expand(obj->get_bbox());
  }
  return bbox;
}

void Scene::set_draw_styles(DrawStyle *defaultStyle, DrawStyle *hoveredStyle,
                             DrawStyle *selectedStyle) {
  for (SceneObject *obj : objects) {
    obj->set_draw_styles(defaultStyle, hoveredStyle, selectedStyle);
  }
}

void Scene::render_in_opengl() {
  for (SceneObject *obj : objects) {
    obj->render_in_opengl();
  }
}


void Scene::update_selection(const Vector2D& p, const Matrix4x4& worldTo3DH) {
  double minW = -1.0;
  int minI = -1;
  for (int i = 0; i < objects.size(); i++) {
    double newW = objects[i]->test_selection(p, worldTo3DH, minW);
    if (newW >= 0.0 && (minI < 0 || newW < minW)) {
      minI = i;
      minW = newW;
    }
  }
  if (minI == -1) {
    for (SceneObject *obj : objects) {
      obj->invalidate_hover();
    }
  } else {
    for (int i = 0; i < minI; i++) {
      objects[i]->invalidate_hover();
    }
    objects[minI]->confirm_hover();
    for (int i = minI + 1; i < objects.size(); i++) {
      objects[i]->invalidate_hover();
    }
  }
  hoverIdx = minI;
}

bool Scene::has_selection() {
  return selectionIdx >= 0;
}

bool Scene::has_hover() {
  return hoverIdx >= 0;
}

void Scene::confirm_selection() {
  if (!has_hover()) return;
  if (has_selection() && selectionIdx != hoverIdx) {
    objects[selectionIdx]->invalidate_selection();
  }
  selectionIdx = hoverIdx;
  objects[selectionIdx]->confirm_select();
}

void Scene::invalidate_selection() {
  if (!has_selection()) return;
  objects[selectionIdx]->invalidate_selection();
  selectionIdx = -1;
}

void Scene::drag_selection(float dx, float dy, const Matrix4x4& worldTo3DH) {
  if (!has_selection()) return;
  objects[selectionIdx]->drag_selection(dx, dy, worldTo3DH);
}

SelectionInfo *Scene::get_selection_info() {
  if (!has_selection()) return nullptr;
  selectionInfo.info.clear();
  objects[selectionIdx]->get_selection_info(&selectionInfo);
  return &selectionInfo;
}

void Scene::collapse_selected_edge() {
  MeshView *meshView = get_selection_as_mesh();
  if (meshView == nullptr) return;
  meshView->collapse_selected_edge();
  invalidate_selection();
}

void Scene::flip_selected_edge() {
  MeshView *meshView = get_selection_as_mesh();
  if (meshView == nullptr) return;
  meshView->flip_selected_edge();
  invalidate_selection();
}

void Scene::split_selected_edge() {
  MeshView *meshView = get_selection_as_mesh();
  if (meshView == nullptr) return;
  meshView->split_selected_edge();
  invalidate_selection();
}

void Scene::upsample_selected_mesh() {
  MeshView *meshView = get_selection_as_mesh();
  if (meshView == nullptr) return;
  meshView->upsample();
  invalidate_selection();
}

void Scene::downsample_selected_mesh() {
  MeshView *meshView = get_selection_as_mesh();
  if (meshView == nullptr) return;
  meshView->downsample();
  invalidate_selection();
}

void Scene::resample_selected_mesh() {
  MeshView *meshView = get_selection_as_mesh();
  if (meshView == nullptr) return;
  meshView->resample();
  invalidate_selection();
}

SceneObjects::Scene *Scene::get_static_scene() {
  std::vector<SceneObjects::SceneObject *> staticObjects;
  std::vector<SceneObjects::SceneLight *> staticLights;

  for (SceneObject *obj : objects) {
    staticObjects.push_back(obj->get_static_object());
  }
  for (SceneLight *light : lights) {
    staticLights.push_back(light->get_static_light());
  }

  return new SceneObjects::Scene(staticObjects, staticLights);
}

std::vector<SceneObject *> Scene::get_wc_objects() {
  std::vector<SceneObject *> ret;
  for (SceneObject *obj : objects) {
    //NOTE: change this conditional to allow more than just meshes to get watercolored.
    //we would likely want a new interface for "watercolorable" objects
    // that defines the various functions needed to (a) simulate and (b) render simulated
    // objects.
    if (dynamic_cast<GLScene::Mesh*>(obj) && obj->get_name().compare("Mesh") == 0) { 
      cout << "adding " << obj->get_name() << " to watercolor objects" << endl;
      ret.push_back(obj);
    }
  }
  return ret;
}

MeshView *Scene::get_selection_as_mesh() {
  SceneObject *selection = get_selection();
  if (selection == nullptr) return nullptr;
  return selection->get_mesh_view();
}

SceneObject *Scene::get_selection() {
  if (!has_selection()) return nullptr;
  return objects[selectionIdx];
}

} // namespace GLScene
} // namespace CGL
