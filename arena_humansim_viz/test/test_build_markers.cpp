#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <arena_humansim_msgs/msg/agent_state.hpp>
#include <arena_humansim_msgs/msg/agent_viz.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "arena_humansim_viz/build_markers.hpp"

namespace
{
using arena_humansim_msgs::msg::AgentState;
using arena_humansim_msgs::msg::AgentViz;
using arena_humansim_viz::KeySet;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

constexpr double kPi = 3.14159265358979323846;

struct Agent
{
  int32_t id;
  double x;
  double y;
  double theta;
  double vx = 0.0;
  double vy = 0.0;
  double radius = 0.3;
  uint8_t kind = AgentState::KIND_HUMAN;
  double vision_range = 2.0;
  double vision_fov = 90.0;
  double proximity_sense = 0.5;
};

AgentViz make_viz(uint8_t level, const std::vector<Agent> & agents)
{
  AgentViz v;
  v.header.stamp.sec = 12;
  v.header.stamp.nanosec = 34;
  v.level = level;
  for (const auto & a : agents) {
    v.agent_id.push_back(a.id);
    v.x.push_back(a.x);
    v.y.push_back(a.y);
    v.theta.push_back(a.theta);
    v.vx.push_back(a.vx);
    v.vy.push_back(a.vy);
    v.radius.push_back(a.radius);
    v.kind.push_back(a.kind);
    if (level >= 2) {
      v.vision_range.push_back(a.vision_range);
      v.vision_fov.push_back(a.vision_fov);
      v.proximity_sense.push_back(a.proximity_sense);
    }
  }
  return v;
}

struct Builder
{
  KeySet keys[2];
  int current = 0;
  MarkerArray out;

  const MarkerArray & build(const AgentViz & v)
  {
    arena_humansim_viz::build_markers(v, keys[current ^ 1], keys[current], out);
    current ^= 1;
    return out;
  }
};

const Marker * find(const MarkerArray & ma, const std::string & ns, int32_t id)
{
  for (const auto & m : ma.markers) {
    if (m.ns == ns && m.id == id && m.action == Marker::ADD) {
      return &m;
    }
  }
  return nullptr;
}

size_t count_ns(const MarkerArray & ma, const std::string & ns)
{
  size_t n = 0;
  for (const auto & m : ma.markers) {
    n += m.ns == ns && m.action == Marker::ADD;
  }
  return n;
}

void expect_color(const Marker & m, double r, double g, double b, double a)
{
  EXPECT_FLOAT_EQ(m.color.r, static_cast<float>(r));
  EXPECT_FLOAT_EQ(m.color.g, static_cast<float>(g));
  EXPECT_FLOAT_EQ(m.color.b, static_cast<float>(b));
  EXPECT_FLOAT_EQ(m.color.a, static_cast<float>(a));
}

void expect_point(const geometry_msgs::msg::Point & p, double x, double y, double z)
{
  EXPECT_DOUBLE_EQ(p.x, x);
  EXPECT_DOUBLE_EQ(p.y, y);
  EXPECT_DOUBLE_EQ(p.z, z);
}

TEST(BuildMarkers, BodyAndHeadingForHumanAndRobot)
{
  Agent human{1, 1.0, 2.0, kPi / 2.0};
  Agent robot{2, 3.0, 4.0, 0.0};
  robot.radius = 0.5;
  robot.kind = AgentState::KIND_ROBOT;
  Builder b;
  const auto & ma = b.build(make_viz(1, {human, robot}));

  const Marker * body = find(ma, "agent_body", 1);
  ASSERT_NE(body, nullptr);
  EXPECT_EQ(body->type, Marker::CYLINDER);
  EXPECT_EQ(body->header.frame_id, "map");
  EXPECT_EQ(body->header.stamp.sec, 12);
  EXPECT_EQ(body->header.stamp.nanosec, 34u);
  EXPECT_DOUBLE_EQ(body->pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(body->pose.position.y, 2.0);
  EXPECT_DOUBLE_EQ(body->pose.position.z, 0.85);
  EXPECT_DOUBLE_EQ(body->pose.orientation.z, std::sin(kPi / 4.0));
  EXPECT_DOUBLE_EQ(body->pose.orientation.w, std::cos(kPi / 4.0));
  EXPECT_DOUBLE_EQ(body->scale.x, 0.6);
  EXPECT_DOUBLE_EQ(body->scale.y, 0.6);
  EXPECT_DOUBLE_EQ(body->scale.z, 1.7);
  expect_color(*body, 0.95, 0.75, 0.55, 0.9);

  const Marker * head = find(ma, "agent_heading", 1);
  ASSERT_NE(head, nullptr);
  EXPECT_EQ(head->type, Marker::ARROW);
  EXPECT_DOUBLE_EQ(head->pose.orientation.w, 1.0);
  EXPECT_DOUBLE_EQ(head->scale.x, 0.03);
  EXPECT_DOUBLE_EQ(head->scale.y, 0.06);
  EXPECT_DOUBLE_EQ(head->scale.z, 0.06);
  expect_color(*head, 0.1, 0.1, 0.1, 0.9);
  ASSERT_EQ(head->points.size(), 2u);
  expect_point(head->points[0], 1.0, 2.0, 1.75);
  expect_point(head->points[1], 1.0 + 0.5 * std::cos(kPi / 2.0), 2.0 + 0.5 * std::sin(kPi / 2.0),
      1.75);

  const Marker * rbody = find(ma, "agent_body", 2);
  ASSERT_NE(rbody, nullptr);
  EXPECT_DOUBLE_EQ(rbody->pose.position.z, 0.25);
  EXPECT_DOUBLE_EQ(rbody->pose.orientation.z, 0.0);
  EXPECT_DOUBLE_EQ(rbody->pose.orientation.w, 1.0);
  EXPECT_DOUBLE_EQ(rbody->scale.x, 1.0);
  EXPECT_DOUBLE_EQ(rbody->scale.z, 0.5);
  expect_color(*rbody, 0.4, 0.4, 0.5, 0.9);

  const Marker * rhead = find(ma, "agent_heading", 2);
  ASSERT_NE(rhead, nullptr);
  expect_point(rhead->points[0], 3.0, 4.0, 0.55);
  expect_point(rhead->points[1], 3.7, 4.0, 0.55);
}

TEST(BuildMarkers, NeedsLabelTextAndColorCycling)
{
  auto v = make_viz(1, {{1, 1.0, 2.0, 0.0}});
  v.need_names = {"hunger", "social"};
  v.need_agent_id = {1, 1, 99};
  v.need_name_idx = {0, 1, 0};
  v.need_slot = {1, 5, 0};
  v.need_value = {42.4f, 99.6f, 10.0f};
  Builder b;
  const auto & ma = b.build(v);

  const Marker * bar = find(ma, "needs_hunger", 1);
  ASSERT_NE(bar, nullptr);
  EXPECT_EQ(bar->type, Marker::CUBE);
  EXPECT_DOUBLE_EQ(bar->scale.x, 0.3 * static_cast<double>(42.4f) / 100.0);
  EXPECT_DOUBLE_EQ(bar->scale.y, 0.05);
  EXPECT_DOUBLE_EQ(bar->scale.z, 0.05);
  EXPECT_DOUBLE_EQ(bar->pose.position.x, 1.25);
  EXPECT_DOUBLE_EQ(bar->pose.position.y, 2.0 + 0.15 * 1 - 0.15);
  EXPECT_DOUBLE_EQ(bar->pose.position.z, 0.7);
  expect_color(*bar, 0.8, 0.8, 0.2, 0.8);

  const Marker * label = find(ma, "needs_hunger_label", 1);
  ASSERT_NE(label, nullptr);
  EXPECT_EQ(label->type, Marker::TEXT_VIEW_FACING);
  EXPECT_EQ(label->text, "hunger:42");
  EXPECT_DOUBLE_EQ(label->scale.z, 0.1);
  EXPECT_DOUBLE_EQ(label->pose.position.x, 1.5);
  expect_color(*label, 0.8, 0.8, 0.2, 0.8);

  const Marker * social = find(ma, "needs_social_label", 1);
  ASSERT_NE(social, nullptr);
  EXPECT_EQ(social->text, "social:100");
  EXPECT_DOUBLE_EQ(social->pose.position.y, 2.0 + 0.15 * 5 - 0.15);
  expect_color(*social, 0.2, 0.8, 0.2, 0.8);
  expect_color(*find(ma, "needs_social", 1), 0.2, 0.8, 0.2, 0.8);

  EXPECT_EQ(count_ns(ma, "needs_hunger"), 1u);
}

TEST(BuildMarkers, NeedBarHasMinimumLength)
{
  auto v = make_viz(1, {{1, 0.0, 0.0, 0.0}});
  v.need_names = {"rest"};
  v.need_agent_id = {1};
  v.need_name_idx = {0};
  v.need_slot = {0};
  v.need_value = {0.0f};
  Builder b;
  const auto & ma = b.build(v);
  EXPECT_DOUBLE_EQ(find(ma, "needs_rest", 1)->scale.x, 0.01);
  EXPECT_EQ(find(ma, "needs_rest_label", 1)->text, "rest:0");
}

TEST(BuildMarkers, CmdLabelsPassThroughLabelTable)
{
  auto v = make_viz(1, {{1, 1.0, 2.0, 0.0}, {2, 3.0, 4.0, 0.0}});
  v.cmd_labels = {"NAVIGATE", "INTR"};
  v.cmd_agent_id = {1, 2, 99, 1};
  v.cmd_label_idx = {1, 0, 0, 7};
  Builder b;
  const auto & ma = b.build(v);

  const Marker * c1 = find(ma, "cmd", 1);
  ASSERT_NE(c1, nullptr);
  EXPECT_EQ(c1->type, Marker::TEXT_VIEW_FACING);
  EXPECT_EQ(c1->text, "INTR");
  EXPECT_DOUBLE_EQ(c1->pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(c1->pose.position.y, 2.0);
  EXPECT_DOUBLE_EQ(c1->pose.position.z, 0.9);
  EXPECT_DOUBLE_EQ(c1->scale.z, 0.2);
  expect_color(*c1, 1.0, 1.0, 1.0, 0.9);
  EXPECT_EQ(find(ma, "cmd", 2)->text, "NAVIGATE");
  EXPECT_EQ(count_ns(ma, "cmd"), 2u);
}

TEST(BuildMarkers, InteractionLinksSkipAbsentParticipant)
{
  auto v = make_viz(1, {{1, 0.0, 0.0, 0.0}, {2, 2.0, 4.0, 0.0}});
  v.interaction_id = {5, 6};
  v.interaction_label = {"chat [3p +1q]", "queue [2p]"};
  v.interaction_offset = {0, 3, 5};
  v.interaction_participants = {1, 99, 2, 98, 97};
  Builder b;
  const auto & ma = b.build(v);

  const Marker * links = find(ma, "interaction_links", 5);
  ASSERT_NE(links, nullptr);
  EXPECT_EQ(links->type, Marker::LINE_LIST);
  EXPECT_DOUBLE_EQ(links->scale.x, 0.03);
  expect_color(*links, 0.9, 0.9, 0.2, 0.6);
  ASSERT_EQ(links->points.size(), 2u);
  expect_point(links->points[0], 0.0, 0.0, 0.3);
  expect_point(links->points[1], 2.0, 4.0, 0.3);

  const Marker * label = find(ma, "interaction_label", 5);
  ASSERT_NE(label, nullptr);
  EXPECT_EQ(label->text, "chat [3p +1q]");
  EXPECT_DOUBLE_EQ(label->pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(label->pose.position.y, 2.0);
  EXPECT_DOUBLE_EQ(label->pose.position.z, 0.5);
  EXPECT_DOUBLE_EQ(label->scale.z, 0.2);
  expect_color(*label, 1.0, 1.0, 0.5, 0.9);

  const Marker * empty_links = find(ma, "interaction_links", 6);
  ASSERT_NE(empty_links, nullptr);
  EXPECT_TRUE(empty_links->points.empty());
  EXPECT_EQ(find(ma, "interaction_label", 6), nullptr);
}

TEST(BuildMarkers, ConeAndProximityTriangles)
{
  Builder b;
  const auto & ma = b.build(make_viz(2, {{1, 1.0, 1.0, 0.0}}));

  const Marker * cone = find(ma, "vision_cone", 1);
  ASSERT_NE(cone, nullptr);
  EXPECT_EQ(cone->type, Marker::TRIANGLE_LIST);
  EXPECT_DOUBLE_EQ(cone->scale.x, 1.0);
  expect_color(*cone, 0.2, 0.6, 1.0, 0.12);
  ASSERT_EQ(cone->points.size(), 36u);
  const double half = kPi / 4.0;
  const double a1 = -half + 2.0 * half / 12;
  expect_point(cone->points[0], 1.0, 1.0, 0.02);
  EXPECT_NEAR(cone->points[1].x, 1.0 + 2.0 * std::cos(-half), 1e-12);
  EXPECT_NEAR(cone->points[1].y, 1.0 + 2.0 * std::sin(-half), 1e-12);
  EXPECT_NEAR(cone->points[2].x, 1.0 + 2.0 * std::cos(a1), 1e-12);
  EXPECT_NEAR(cone->points[2].y, 1.0 + 2.0 * std::sin(a1), 1e-12);
  EXPECT_DOUBLE_EQ(cone->points[2].z, 0.02);

  const Marker * prox = find(ma, "proximity_sense", 1);
  ASSERT_NE(prox, nullptr);
  EXPECT_EQ(prox->type, Marker::TRIANGLE_LIST);
  expect_color(*prox, 0.2, 0.6, 1.0, 0.12);
  ASSERT_EQ(prox->points.size(), 72u);
  expect_point(prox->points[0], 1.0, 1.0, 0.02);
  expect_point(prox->points[1], 1.5, 1.0, 0.02);
  EXPECT_NEAR(prox->points[2].x, 1.0 + 0.5 * std::cos(2.0 * kPi / 24), 1e-12);
  EXPECT_NEAR(prox->points[2].y, 1.0 + 0.5 * std::sin(2.0 * kPi / 24), 1e-12);
}

TEST(BuildMarkers, ObservedIdCombinesAgentAndSlot)
{
  auto v = make_viz(2, {{7, 1.0, 2.0, 0.0}});
  v.observed_agent_id = {7, 99};
  v.observed_slot = {3, 0};
  v.observed_x = {5.0, 0.0};
  v.observed_y = {6.0, 0.0};
  Builder b;
  const auto & ma = b.build(v);

  const Marker * obs = find(ma, "observed", 703);
  ASSERT_NE(obs, nullptr);
  EXPECT_EQ(obs->type, Marker::ARROW);
  EXPECT_DOUBLE_EQ(obs->scale.x, 0.015);
  EXPECT_DOUBLE_EQ(obs->scale.y, 0.03);
  EXPECT_DOUBLE_EQ(obs->scale.z, 0.03);
  expect_color(*obs, 0.2, 0.8, 0.2, 0.4);
  expect_point(obs->points[0], 1.0, 2.0, 0.1);
  expect_point(obs->points[1], 5.0, 6.0, 0.1);
  EXPECT_EQ(count_ns(ma, "observed"), 1u);
}

TEST(BuildMarkers, PathAndGoals)
{
  auto v = make_viz(2, {{1, 0.0, 0.0, 0.0}, {2, 1.0, 1.0, 0.0}});
  v.path_agent_id = {1, 2};
  v.path_offset = {0, 3, 4};
  v.path_x = {0.0, 1.0, 2.0, 9.0};
  v.path_y = {0.0, 0.5, 1.0, 9.0};
  v.igoal_agent_id = {1, 99};
  v.igoal_x = {1.0, 0.0};
  v.igoal_y = {0.5, 0.0};
  v.goal_agent_id = {1};
  v.goal_x = {2.0};
  v.goal_y = {1.0};
  v.goal_theta = {kPi};
  Builder b;
  const auto & ma = b.build(v);

  const Marker * path = find(ma, "path", 1);
  ASSERT_NE(path, nullptr);
  EXPECT_EQ(path->type, Marker::LINE_STRIP);
  EXPECT_DOUBLE_EQ(path->scale.x, 0.02);
  expect_color(*path, 0.4, 0.9, 0.4, 0.5);
  ASSERT_EQ(path->points.size(), 3u);
  expect_point(path->points[2], 2.0, 1.0, 0.05);
  EXPECT_EQ(find(ma, "path", 2), nullptr);

  const Marker * igoal = find(ma, "igoal", 1);
  ASSERT_NE(igoal, nullptr);
  EXPECT_EQ(igoal->type, Marker::SPHERE);
  EXPECT_DOUBLE_EQ(igoal->scale.x, 0.2);
  EXPECT_DOUBLE_EQ(igoal->pose.position.z, 0.1);
  expect_color(*igoal, 0.1, 1.0, 0.1, 0.8);
  EXPECT_EQ(count_ns(ma, "igoal"), 1u);

  const Marker * goal = find(ma, "goal", 1);
  ASSERT_NE(goal, nullptr);
  EXPECT_DOUBLE_EQ(goal->scale.x, 0.05);
  EXPECT_DOUBLE_EQ(goal->scale.y, 0.1);
  EXPECT_DOUBLE_EQ(goal->scale.z, 0.08);
  expect_color(*goal, 1.0, 0.3, 0.3, 0.8);
  expect_point(goal->points[0], 2.0, 1.0, 0.1);
  expect_point(goal->points[1], 2.0 + 0.3 * std::cos(kPi), 1.0 + 0.3 * std::sin(kPi), 0.1);
}

TEST(BuildMarkers, WaypointActiveStylingAndRadiusSkip)
{
  auto v = make_viz(2, {{1, 0.0, 0.0, 0.0}, {2, 0.0, 0.0, 0.0}});
  v.wp_agent_id = {1, 2};
  v.wp_offset = {0, 3, 4};
  v.wp_x = {1.0, 2.0, 3.0, 7.0};
  v.wp_y = {1.5, 2.5, 3.5, 8.0};
  v.wp_active = {1, 0};
  v.wp_active_radius = {0.0, 0.4};
  Builder b;
  const auto & ma = b.build(v);

  const Marker * path = find(ma, "wp_path", 1);
  ASSERT_NE(path, nullptr);
  EXPECT_EQ(path->type, Marker::LINE_STRIP);
  ASSERT_EQ(path->points.size(), 3u);
  expect_color(*path, 0.7, 0.5, 1.0, 0.5);
  expect_point(path->points[0], 1.0, 1.5, 0.05);

  const Marker * active = find(ma, "wp", 101);
  ASSERT_NE(active, nullptr);
  EXPECT_EQ(active->type, Marker::SPHERE);
  EXPECT_DOUBLE_EQ(active->scale.x, 0.2);
  expect_color(*active, 1.0, 0.5, 1.0, 0.8);
  EXPECT_DOUBLE_EQ(active->pose.position.x, 2.0);
  EXPECT_DOUBLE_EQ(active->pose.position.y, 2.5);
  EXPECT_DOUBLE_EQ(active->pose.position.z, 0.1);
  const Marker * idle = find(ma, "wp", 102);
  ASSERT_NE(idle, nullptr);
  EXPECT_DOUBLE_EQ(idle->scale.x, 0.12);
  expect_color(*idle, 0.7, 0.5, 1.0, 0.5);
  EXPECT_NE(find(ma, "wp", 100), nullptr);
  EXPECT_EQ(find(ma, "wp_rad", 1), nullptr);

  EXPECT_EQ(find(ma, "wp_path", 2), nullptr);
  const Marker * rad = find(ma, "wp_rad", 2);
  ASSERT_NE(rad, nullptr);
  EXPECT_EQ(rad->type, Marker::CYLINDER);
  EXPECT_DOUBLE_EQ(rad->scale.x, 0.8);
  EXPECT_DOUBLE_EQ(rad->scale.y, 0.8);
  EXPECT_DOUBLE_EQ(rad->scale.z, 0.02);
  EXPECT_DOUBLE_EQ(rad->pose.position.x, 7.0);
  EXPECT_DOUBLE_EQ(rad->pose.position.y, 8.0);
  EXPECT_DOUBLE_EQ(rad->pose.position.z, 0.01);
  expect_color(*rad, 0.7, 0.5, 1.0, 0.15);
  expect_color(*find(ma, "wp", 200), 1.0, 0.5, 1.0, 0.8);
}

TEST(BuildMarkers, VelocitySkippedBelowThreshold)
{
  Agent slow{1, 0.0, 0.0, 0.0};
  slow.vx = 5e-5;
  slow.vy = -5e-5;
  Agent fast{2, 1.0, 2.0, 0.0};
  fast.vx = 1e-4;
  fast.vy = 0.5;
  Builder b;
  const auto & ma = b.build(make_viz(2, {slow, fast}));

  EXPECT_EQ(find(ma, "vel", 1), nullptr);
  const Marker * vel = find(ma, "vel", 2);
  ASSERT_NE(vel, nullptr);
  EXPECT_EQ(vel->type, Marker::ARROW);
  EXPECT_DOUBLE_EQ(vel->scale.x, 0.03);
  expect_color(*vel, 0.0, 0.7, 1.0, 0.9);
  expect_point(vel->points[0], 1.0, 2.0, 0.1);
  expect_point(vel->points[1], 1.0 + 1e-4, 2.5, 0.1);
}

TEST(BuildMarkers, StaleKeysDeletedBeforeAdds)
{
  Agent a{1, 0.0, 0.0, 0.0};
  a.vx = 1.0;
  Agent c{2, 1.0, 1.0, 0.0};
  c.vx = 1.0;
  Builder b;
  auto first = make_viz(2, {a, c});
  first.cmd_labels = {"WAIT"};
  first.cmd_agent_id = {2};
  first.cmd_label_idx = {0};
  b.build(first);

  a.vx = 0.0;
  const auto & ma = b.build(make_viz(2, {a}));

  std::set<std::pair<std::string, int32_t>> deleted;
  bool seen_add = false;
  for (const auto & m : ma.markers) {
    if (m.action == Marker::DELETE) {
      EXPECT_FALSE(seen_add);
      EXPECT_EQ(m.header.frame_id, "map");
      EXPECT_EQ(m.header.stamp.sec, 12);
      EXPECT_TRUE(m.points.empty());
      EXPECT_TRUE(m.text.empty());
      deleted.emplace(m.ns, m.id);
    } else {
      seen_add = true;
    }
  }
  const std::set<std::pair<std::string, int32_t>> expected{
    {"agent_body", 2}, {"agent_heading", 2}, {"cmd", 2}, {"vision_cone", 2},
    {"proximity_sense", 2}, {"vel", 1}, {"vel", 2}};
  EXPECT_EQ(deleted, expected);
  EXPECT_NE(find(ma, "agent_body", 1), nullptr);
  EXPECT_EQ(ma.markers.size(), expected.size() + 4u);
}

TEST(BuildMarkers, LevelZeroClearsEverything)
{
  auto v = make_viz(2, {{1, 0.0, 0.0, 0.0}, {2, 1.0, 1.0, 0.0}});
  v.need_names = {"hunger"};
  v.need_agent_id = {1};
  v.need_name_idx = {0};
  v.need_slot = {0};
  v.need_value = {50.0f};
  Builder b;
  const size_t drawn = b.build(v).markers.size();
  ASSERT_GT(drawn, 0u);

  const auto & cleared = b.build(make_viz(0, {}));
  EXPECT_EQ(cleared.markers.size(), drawn);
  for (const auto & m : cleared.markers) {
    EXPECT_EQ(m.action, Marker::DELETE);
  }

  EXPECT_TRUE(b.build(make_viz(0, {})).markers.empty());
}

TEST(BuildMarkers, LevelOneSkipsLevelTwoNamespaces)
{
  Agent a{1, 0.0, 0.0, 0.0};
  a.vx = 1.0;
  auto v = make_viz(1, {a});
  v.vision_range = {2.0};
  v.vision_fov = {90.0};
  v.proximity_sense = {0.5};
  v.observed_agent_id = {1};
  v.observed_slot = {0};
  v.observed_x = {1.0};
  v.observed_y = {1.0};
  v.path_agent_id = {1};
  v.path_offset = {0, 2};
  v.path_x = {0.0, 1.0};
  v.path_y = {0.0, 1.0};
  v.igoal_agent_id = {1};
  v.igoal_x = {1.0};
  v.igoal_y = {1.0};
  v.goal_agent_id = {1};
  v.goal_x = {1.0};
  v.goal_y = {1.0};
  v.goal_theta = {0.0};
  v.wp_agent_id = {1};
  v.wp_offset = {0, 2};
  v.wp_x = {0.0, 1.0};
  v.wp_y = {0.0, 1.0};
  v.wp_active = {0};
  v.wp_active_radius = {0.3};
  Builder b;
  const auto & ma = b.build(v);

  const std::set<std::string> level_two{
    "vision_cone", "proximity_sense", "observed", "path", "igoal", "goal", "vel",
    "wp_path", "wp", "wp_rad"};
  for (const auto & m : ma.markers) {
    EXPECT_EQ(level_two.count(m.ns), 0u) << m.ns;
  }
  EXPECT_EQ(ma.markers.size(), 2u);
}

}  // namespace

TEST(BuildMarkers, ShiftMovesAddPosesAndKeepsPoints)
{
  Builder b;
  Agent a{1, 1.0, 2.0, 0.0};
  a.vx = 1.0;
  b.build(make_viz(2, {a}));
  MarkerArray ma = b.build(make_viz(1, {a}));
  arena_humansim_viz::shift_markers(ma, 10.0, -5.0);

  const Marker * body = find(ma, "agent_body", 1);
  ASSERT_NE(body, nullptr);
  EXPECT_DOUBLE_EQ(body->pose.position.x, 11.0);
  EXPECT_DOUBLE_EQ(body->pose.position.y, -3.0);
  const Marker * heading = find(ma, "agent_heading", 1);
  ASSERT_NE(heading, nullptr);
  EXPECT_DOUBLE_EQ(heading->pose.position.x, 10.0);
  EXPECT_DOUBLE_EQ(heading->pose.position.y, -5.0);
  expect_point(heading->points[0], 1.0, 2.0, 1.75);
  size_t deletes = 0;
  for (const auto & m : ma.markers) {
    if (m.action == Marker::DELETE) {
      ++deletes;
      EXPECT_DOUBLE_EQ(m.pose.position.x, 0.0);
      EXPECT_DOUBLE_EQ(m.pose.position.y, 0.0);
    }
  }
  EXPECT_GT(deletes, 0u);
}
