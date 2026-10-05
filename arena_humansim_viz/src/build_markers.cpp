#include "arena_humansim_viz/build_markers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <utility>

#include <arena_humansim_msgs/msg/agent_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/color_rgba.hpp>

namespace arena_humansim_viz
{
namespace
{
using AgentViz = arena_humansim_msgs::msg::AgentViz;
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;
using ColorRGBA = std_msgs::msg::ColorRGBA;
using Ids = std::vector<int32_t>;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr int kConeSegments = 12;
constexpr int kProximitySegments = 24;

ColorRGBA rgba(double r, double g, double b, double a)
{
  ColorRGBA c;
  c.r = static_cast<float>(r);
  c.g = static_cast<float>(g);
  c.b = static_cast<float>(b);
  c.a = static_cast<float>(a);
  return c;
}

const ColorRGBA kCone = rgba(0.2, 0.6, 1.0, 0.12);
const ColorRGBA kProx = kCone;
const ColorRGBA kObs = rgba(0.2, 0.8, 0.2, 0.4);
const ColorRGBA kCmd = rgba(1.0, 1.0, 1.0, 0.9);
const ColorRGBA kPath = rgba(0.4, 0.9, 0.4, 0.5);
const ColorRGBA kIgoal = rgba(0.1, 1.0, 0.1, 0.8);
const ColorRGBA kGoal = rgba(1.0, 0.3, 0.3, 0.8);
const ColorRGBA kVel = rgba(0.0, 0.7, 1.0, 0.9);
const ColorRGBA kIlink = rgba(0.9, 0.9, 0.2, 0.6);
const ColorRGBA kIlabel = rgba(1.0, 1.0, 0.5, 0.9);
const ColorRGBA kWp = rgba(0.7, 0.5, 1.0, 0.5);
const ColorRGBA kWpAct = rgba(1.0, 0.5, 1.0, 0.8);
const ColorRGBA kWpRad = rgba(0.7, 0.5, 1.0, 0.15);
const ColorRGBA kBodyHuman = rgba(0.95, 0.75, 0.55, 0.9);
const ColorRGBA kBodyRobot = rgba(0.4, 0.4, 0.5, 0.9);
const ColorRGBA kHeading = rgba(0.1, 0.1, 0.1, 0.9);
const std::array<ColorRGBA, 5> kNeedColors{
  rgba(0.2, 0.8, 0.2, 0.8),
  rgba(0.8, 0.8, 0.2, 0.8),
  rgba(0.2, 0.6, 1.0, 0.8),
  rgba(0.9, 0.4, 0.1, 0.8),
  rgba(0.8, 0.2, 0.8, 0.8),
};

const std::string kFrame = "map";
const std::string kNsBody = "agent_body";
const std::string kNsHeading = "agent_heading";
const std::string kNsCmd = "cmd";
const std::string kNsLinks = "interaction_links";
const std::string kNsLabel = "interaction_label";
const std::string kNsCone = "vision_cone";
const std::string kNsProx = "proximity_sense";
const std::string kNsObserved = "observed";
const std::string kNsPath = "path";
const std::string kNsIgoal = "igoal";
const std::string kNsGoal = "goal";
const std::string kNsVel = "vel";
const std::string kNsWpPath = "wp_path";
const std::string kNsWp = "wp";
const std::string kNsWpRad = "wp_rad";

class Emitter
{
public:
  Emitter(MarkerArray & out, const builtin_interfaces::msg::Time & stamp)
  : out_(out), stamp_(stamp) {}

  Marker & add(const std::string & ns, Ids & keys, int32_t id, int32_t type)
  {
    Marker & m = reset(ns, id, Marker::ADD);
    m.type = type;
    keys.push_back(id);
    return m;
  }

  void remove(const std::string & ns, int32_t id)
  {
    Marker & m = reset(ns, id, Marker::DELETE);
    m.type = Marker::ARROW;
  }

  size_t size() const {return n_;}

private:
  Marker & reset(const std::string & ns, int32_t id, int32_t action)
  {
    if (n_ == out_.markers.size()) {
      out_.markers.emplace_back();
    }
    Marker & m = out_.markers[n_++];
    m.header.frame_id = kFrame;
    m.header.stamp = stamp_;
    m.ns = ns;
    m.id = id;
    m.action = action;
    m.pose = geometry_msgs::msg::Pose();
    m.scale = geometry_msgs::msg::Vector3();
    m.color = ColorRGBA();
    m.points.clear();
    m.text.clear();
    return m;
  }

  MarkerArray & out_;
  const builtin_interfaces::msg::Time & stamp_;
  size_t n_ = 0;
};

void add_point(Marker & m, double x, double y, double z)
{
  auto & p = m.points.emplace_back();
  p.x = x;
  p.y = y;
  p.z = z;
}

void set_scale(Marker & m, double x, double y, double z)
{
  m.scale.x = x;
  m.scale.y = y;
  m.scale.z = z;
}

void set_position(Marker & m, double x, double y, double z)
{
  m.pose.position.x = x;
  m.pose.position.y = y;
  m.pose.position.z = z;
}

void set_arrow(Marker & m, double x0, double y0, double z0, double x1, double y1, double z1)
{
  add_point(m, x0, y0, z0);
  add_point(m, x1, y1, z1);
}

int32_t slot_id(int32_t aid, int64_t slot)
{
  return static_cast<int32_t>(static_cast<int64_t>(aid) * 100 + slot);
}

bool span(
  const std::vector<uint32_t> & offset, size_t k, size_t data_size, size_t & begin,
  size_t & end)
{
  if (k + 1 >= offset.size()) {
    return false;
  }
  begin = offset[k];
  end = offset[k + 1];
  return begin <= end && end <= data_size;
}

class AgentIndex
{
public:
  explicit AgentIndex(const std::vector<int32_t> & ids, size_t n)
  {
    entries_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      entries_.emplace_back(ids[i], i);
    }
    std::stable_sort(
      entries_.begin(), entries_.end(),
      [](const auto & a, const auto & b) {return a.first < b.first;});
  }

  bool find(int32_t id, size_t & index) const
  {
    auto it = std::upper_bound(
      entries_.begin(), entries_.end(), id,
      [](int32_t v, const auto & e) {return v < e.first;});
    if (it == entries_.begin() || std::prev(it)->first != id) {
      return false;
    }
    index = std::prev(it)->second;
    return true;
  }

private:
  std::vector<std::pair<int32_t, size_t>> entries_;
};

size_t agent_count(const AgentViz & v)
{
  return std::min(
    {v.agent_id.size(), v.x.size(), v.y.size(), v.theta.size(), v.vx.size(), v.vy.size(),
      v.radius.size(), v.kind.size()});
}

void draw_agents(const AgentViz & v, size_t n, Emitter & em, KeySet & keys)
{
  Ids & body_ids = keys[kNsBody];
  Ids & head_ids = keys[kNsHeading];
  for (size_t i = 0; i < n; ++i) {
    const int32_t aid = v.agent_id[i];
    const double x = v.x[i];
    const double y = v.y[i];
    const double th = v.theta[i];
    const double radius = v.radius[i];
    const bool is_robot = v.kind[i] == arena_humansim_msgs::msg::AgentState::KIND_ROBOT;
    const double height = is_robot ? 0.5 : 1.7;

    Marker & body = em.add(kNsBody, body_ids, aid, Marker::CYLINDER);
    set_position(body, x, y, height / 2.0);
    body.pose.orientation.z = std::sin(th / 2.0);
    body.pose.orientation.w = std::cos(th / 2.0);
    set_scale(body, radius * 2.0, radius * 2.0, height);
    body.color = is_robot ? kBodyRobot : kBodyHuman;

    Marker & head = em.add(kNsHeading, head_ids, aid, Marker::ARROW);
    set_scale(head, 0.03, 0.06, 0.06);
    head.color = kHeading;
    const double tip_len = radius + 0.2;
    set_arrow(
      head, x, y, height + 0.05,
      x + tip_len * std::cos(th), y + tip_len * std::sin(th), height + 0.05);
  }
}

void draw_cmds(const AgentViz & v, const AgentIndex & index, Emitter & em, KeySet & keys)
{
  Ids & ids = keys[kNsCmd];
  const size_t n = std::min(v.cmd_agent_id.size(), v.cmd_label_idx.size());
  for (size_t k = 0; k < n; ++k) {
    size_t i;
    const size_t label = v.cmd_label_idx[k];
    if (!index.find(v.cmd_agent_id[k], i) || label >= v.cmd_labels.size()) {
      continue;
    }
    Marker & m = em.add(kNsCmd, ids, v.cmd_agent_id[k], Marker::TEXT_VIEW_FACING);
    m.color = kCmd;
    set_position(m, v.x[i], v.y[i], 0.9);
    m.scale.z = 0.2;
    m.text = v.cmd_labels[label];
  }
}

void draw_needs(const AgentViz & v, const AgentIndex & index, Emitter & em, KeySet & keys)
{
  struct NeedNs
  {
    std::string bar;
    std::string label;
    Ids * bar_ids;
    Ids * label_ids;
  };
  std::vector<NeedNs> table;
  table.reserve(v.need_names.size());
  for (const auto & name : v.need_names) {
    NeedNs entry{"needs_" + name, "needs_" + name + "_label", nullptr, nullptr};
    entry.bar_ids = &keys[entry.bar];
    entry.label_ids = &keys[entry.label];
    table.push_back(std::move(entry));
  }

  char buf[64];
  const size_t n = std::min(
    {v.need_agent_id.size(), v.need_name_idx.size(), v.need_slot.size(), v.need_value.size()});
  for (size_t k = 0; k < n; ++k) {
    size_t i;
    const size_t name_idx = v.need_name_idx[k];
    if (!index.find(v.need_agent_id[k], i) || name_idx >= table.size()) {
      continue;
    }
    const int32_t aid = v.need_agent_id[k];
    const NeedNs & ns = table[name_idx];
    const ColorRGBA & clr = kNeedColors[v.need_slot[k] % kNeedColors.size()];
    const double value = v.need_value[k];
    const double y = v.y[i] + 0.15 * v.need_slot[k] - 0.15;

    Marker & bar = em.add(ns.bar, *ns.bar_ids, aid, Marker::CUBE);
    bar.color = clr;
    set_position(bar, v.x[i] + 0.25, y, 0.7);
    set_scale(bar, std::max(0.3 * value / 100.0, 0.01), 0.05, 0.05);

    Marker & label = em.add(ns.label, *ns.label_ids, aid, Marker::TEXT_VIEW_FACING);
    label.color = clr;
    label.scale.z = 0.1;
    set_position(label, v.x[i] + 0.5, y, 0.7);
    std::snprintf(buf, sizeof(buf), ":%.0f", value);
    label.text = v.need_names[name_idx];
    label.text += buf;
  }
}

void draw_interactions(const AgentViz & v, const AgentIndex & index, Emitter & em, KeySet & keys)
{
  Ids & link_ids = keys[kNsLinks];
  Ids & label_ids = keys[kNsLabel];
  const auto & parts = v.interaction_participants;
  const size_t n = std::min(v.interaction_id.size(), v.interaction_label.size());
  for (size_t k = 0; k < n; ++k) {
    size_t begin, end;
    if (!span(v.interaction_offset, k, parts.size(), begin, end)) {
      continue;
    }
    const int32_t iid = v.interaction_id[k];
    if (end - begin >= 2) {
      Marker & m = em.add(kNsLinks, link_ids, iid, Marker::LINE_LIST);
      m.scale.x = 0.03;
      m.color = kIlink;
      for (size_t a = begin; a < end; ++a) {
        for (size_t b = a + 1; b < end; ++b) {
          size_t i, j;
          if (index.find(parts[a], i) && index.find(parts[b], j)) {
            add_point(m, v.x[i], v.y[i], 0.3);
            add_point(m, v.x[j], v.y[j], 0.3);
          }
        }
      }
    }
    double sx = 0.0;
    double sy = 0.0;
    size_t live = 0;
    for (size_t a = begin; a < end; ++a) {
      size_t i;
      if (index.find(parts[a], i)) {
        sx += v.x[i];
        sy += v.y[i];
        ++live;
      }
    }
    if (live > 0) {
      Marker & m = em.add(kNsLabel, label_ids, iid, Marker::TEXT_VIEW_FACING);
      m.color = kIlabel;
      m.scale.z = 0.2;
      set_position(
        m, sx / static_cast<double>(live), sy / static_cast<double>(live), 0.5);
      m.text = v.interaction_label[k];
    }
  }
}

void draw_perception(
  const AgentViz & v, size_t n, const AgentIndex & index, Emitter & em,
  KeySet & keys)
{
  Ids & cone_ids = keys[kNsCone];
  Ids & prox_ids = keys[kNsProx];
  Ids & obs_ids = keys[kNsObserved];
  const size_t np = std::min(
    {n, v.vision_range.size(), v.vision_fov.size(), v.proximity_sense.size()});
  for (size_t i = 0; i < np; ++i) {
    const int32_t aid = v.agent_id[i];
    const double ox = v.x[i];
    const double oy = v.y[i];
    const double z = 0.02;

    Marker & cone = em.add(kNsCone, cone_ids, aid, Marker::TRIANGLE_LIST);
    set_scale(cone, 1.0, 1.0, 1.0);
    cone.color = kCone;
    cone.points.reserve(3 * kConeSegments);
    const double half = std::min(v.vision_fov[i], 360.0) * 0.5 * kDegToRad;
    const double h = v.theta[i];
    const double range = v.vision_range[i];
    for (int si = 0; si < kConeSegments; ++si) {
      const double a0 = h - half + 2.0 * half * si / kConeSegments;
      const double a1 = h - half + 2.0 * half * (si + 1) / kConeSegments;
      add_point(cone, ox, oy, z);
      add_point(cone, ox + range * std::cos(a0), oy + range * std::sin(a0), z);
      add_point(cone, ox + range * std::cos(a1), oy + range * std::sin(a1), z);
    }

    Marker & prox = em.add(kNsProx, prox_ids, aid, Marker::TRIANGLE_LIST);
    set_scale(prox, 1.0, 1.0, 1.0);
    prox.color = kProx;
    prox.points.reserve(3 * kProximitySegments);
    const double radius = v.proximity_sense[i];
    for (int si = 0; si < kProximitySegments; ++si) {
      const double a0 = 2.0 * kPi * si / kProximitySegments;
      const double a1 = 2.0 * kPi * (si + 1) / kProximitySegments;
      add_point(prox, ox, oy, z);
      add_point(prox, ox + radius * std::cos(a0), oy + radius * std::sin(a0), z);
      add_point(prox, ox + radius * std::cos(a1), oy + radius * std::sin(a1), z);
    }
  }

  const size_t no = std::min(
    {v.observed_agent_id.size(), v.observed_slot.size(), v.observed_x.size(),
      v.observed_y.size()});
  for (size_t k = 0; k < no; ++k) {
    size_t i;
    if (!index.find(v.observed_agent_id[k], i)) {
      continue;
    }
    Marker & m = em.add(
      kNsObserved, obs_ids, slot_id(v.observed_agent_id[k], v.observed_slot[k]), Marker::ARROW);
    set_scale(m, 0.015, 0.03, 0.03);
    m.color = kObs;
    set_arrow(m, v.x[i], v.y[i], 0.1, v.observed_x[k], v.observed_y[k], 0.1);
  }
}

void draw_global_plan(const AgentViz & v, const AgentIndex & index, Emitter & em, KeySet & keys)
{
  Ids & path_ids = keys[kNsPath];
  Ids & igoal_ids = keys[kNsIgoal];
  Ids & goal_ids = keys[kNsGoal];
  size_t i;

  const size_t path_points = std::min(v.path_x.size(), v.path_y.size());
  for (size_t k = 0; k < v.path_agent_id.size(); ++k) {
    size_t begin, end;
    if (!index.find(v.path_agent_id[k], i) ||
      !span(v.path_offset, k, path_points, begin, end) || end - begin <= 1)
    {
      continue;
    }
    Marker & m = em.add(kNsPath, path_ids, v.path_agent_id[k], Marker::LINE_STRIP);
    m.scale.x = 0.02;
    m.color = kPath;
    m.points.reserve(end - begin);
    for (size_t p = begin; p < end; ++p) {
      add_point(m, v.path_x[p], v.path_y[p], 0.05);
    }
  }

  const size_t ni = std::min({v.igoal_agent_id.size(), v.igoal_x.size(), v.igoal_y.size()});
  for (size_t k = 0; k < ni; ++k) {
    if (!index.find(v.igoal_agent_id[k], i)) {
      continue;
    }
    Marker & m = em.add(kNsIgoal, igoal_ids, v.igoal_agent_id[k], Marker::SPHERE);
    set_scale(m, 0.1 * 2.0, 0.1 * 2.0, 0.1 * 2.0);
    m.color = kIgoal;
    set_position(m, v.igoal_x[k], v.igoal_y[k], 0.1);
  }

  const size_t ng = std::min(
    {v.goal_agent_id.size(), v.goal_x.size(), v.goal_y.size(), v.goal_theta.size()});
  for (size_t k = 0; k < ng; ++k) {
    if (!index.find(v.goal_agent_id[k], i)) {
      continue;
    }
    Marker & m = em.add(kNsGoal, goal_ids, v.goal_agent_id[k], Marker::ARROW);
    set_scale(m, 0.05, 0.1, 0.08);
    m.color = kGoal;
    const double tx = v.goal_x[k];
    const double ty = v.goal_y[k];
    const double tth = v.goal_theta[k];
    set_arrow(m, tx, ty, 0.1, tx + 0.3 * std::cos(tth), ty + 0.3 * std::sin(tth), 0.1);
  }
}

void draw_velocities(const AgentViz & v, size_t n, Emitter & em, KeySet & keys)
{
  Ids & ids = keys[kNsVel];
  for (size_t i = 0; i < n; ++i) {
    const double vx = v.vx[i];
    const double vy = v.vy[i];
    if (std::abs(vx) < 1e-4 && std::abs(vy) < 1e-4) {
      continue;
    }
    Marker & m = em.add(kNsVel, ids, v.agent_id[i], Marker::ARROW);
    set_scale(m, 0.03, 0.06, 0.06);
    m.color = kVel;
    set_arrow(m, v.x[i], v.y[i], 0.1, v.x[i] + vx, v.y[i] + vy, 0.1);
  }
}

void draw_waypoints(const AgentViz & v, const AgentIndex & index, Emitter & em, KeySet & keys)
{
  Ids & path_ids = keys[kNsWpPath];
  Ids & wp_ids = keys[kNsWp];
  Ids & rad_ids = keys[kNsWpRad];
  const size_t wp_points = std::min(v.wp_x.size(), v.wp_y.size());
  const size_t n = std::min(
    {v.wp_agent_id.size(), v.wp_active.size(), v.wp_active_radius.size()});
  for (size_t k = 0; k < n; ++k) {
    size_t i, begin, end;
    if (!index.find(v.wp_agent_id[k], i) || !span(v.wp_offset, k, wp_points, begin, end) ||
      begin == end)
    {
      continue;
    }
    const int32_t aid = v.wp_agent_id[k];
    const size_t count = end - begin;
    const size_t active = v.wp_active[k];
    if (count > 1) {
      Marker & m = em.add(kNsWpPath, path_ids, aid, Marker::LINE_STRIP);
      m.scale.x = 0.02;
      m.color = kWp;
      m.points.reserve(count);
      for (size_t p = begin; p < end; ++p) {
        add_point(m, v.wp_x[p], v.wp_y[p], 0.05);
      }
    }
    for (size_t w = 0; w < count; ++w) {
      const bool is_active = w == active;
      const double radius = is_active ? 0.1 : 0.06;
      Marker & m = em.add(kNsWp, wp_ids, slot_id(aid, static_cast<int64_t>(w)), Marker::SPHERE);
      m.color = is_active ? kWpAct : kWp;
      set_scale(m, radius * 2.0, radius * 2.0, radius * 2.0);
      set_position(m, v.wp_x[begin + w], v.wp_y[begin + w], 0.1);
    }
    const double r = v.wp_active_radius[k];
    if (r > 0 && active < count) {
      Marker & m = em.add(kNsWpRad, rad_ids, aid, Marker::CYLINDER);
      m.color = kWpRad;
      set_position(m, v.wp_x[begin + active], v.wp_y[begin + active], 0.0 + 0.02 / 2.0);
      set_scale(m, r * 2.0, r * 2.0, 0.02);
    }
  }
}

}  // namespace

void build_markers(
  const AgentViz & viz,
  const KeySet & previous,
  KeySet & current,
  MarkerArray & out)
{
  for (auto & entry : current) {
    entry.second.clear();
  }
  Emitter em(out, viz.header.stamp);

  if (viz.level >= 1) {
    const size_t n = agent_count(viz);
    const AgentIndex index(viz.agent_id, n);
    draw_agents(viz, n, em, current);
    draw_cmds(viz, index, em, current);
    draw_needs(viz, index, em, current);
    draw_interactions(viz, index, em, current);
    if (viz.level >= 2) {
      draw_perception(viz, n, index, em, current);
      draw_global_plan(viz, index, em, current);
      draw_velocities(viz, n, em, current);
      draw_waypoints(viz, index, em, current);
    }
  }

  for (auto & entry : current) {
    Ids & ids = entry.second;
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  }

  const size_t adds = em.size();
  static const Ids kEmpty;
  for (const auto & [ns, prev_ids] : previous) {
    auto it = current.find(ns);
    const Ids & now = it == current.end() ? kEmpty : it->second;
    auto cur = now.begin();
    for (const int32_t id : prev_ids) {
      cur = std::lower_bound(cur, now.end(), id);
      if (cur == now.end() || *cur != id) {
        em.remove(ns, id);
      }
    }
  }

  const size_t total = em.size();
  const size_t swaps = std::min(adds, total - adds);
  for (size_t s = 0; s < swaps; ++s) {
    std::swap(out.markers[s], out.markers[total - swaps + s]);
  }
  out.markers.resize(total);
}

void shift_markers(MarkerArray & markers, double dx, double dy)
{
  for (Marker & m : markers.markers) {
    if (m.action == Marker::ADD) {
      m.pose.position.x += dx;
      m.pose.position.y += dy;
    }
  }
}

}  // namespace arena_humansim_viz
