#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <arena_humansim_msgs/msg/agent_viz.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace arena_humansim_viz
{

/// Marker ids drawn per namespace, sorted and unique.
using KeySet = std::map<std::string, std::vector<int32_t>>;

/// Fills `out` with DELETEs for stale `previous` keys then the ADDs, `current` with drawn keys.
void build_markers(
  const arena_humansim_msgs::msg::AgentViz & viz,
  const KeySet & previous,
  KeySet & current,
  visualization_msgs::msg::MarkerArray & out);

/// Moves every ADD marker's pose by (dx, dy), points stay relative to it.
void shift_markers(visualization_msgs::msg::MarkerArray & markers, double dx, double dy);

}  // namespace arena_humansim_viz
