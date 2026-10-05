# Per-tick marker inputs, published while publish_markers >= 1
# Ragged lists: item i spans [offset[i], offset[i+1]), len(offset) = count + 1
std_msgs/Header header
uint8 level                  # publish_markers level the content covers

# every agent in the pool, own and external
int32[] agent_id
float64[] x
float64[] y
float64[] theta
float64[] vx
float64[] vy
float64[] radius
uint8[] kind                 # AgentState.KIND_*
float64[] vision_range       # level 2, else empty
float64[] vision_fov         # degrees, level 2, else empty
float64[] proximity_sense    # level 2, else empty

# command label per commanded agent
string[] cmd_labels
int32[] cmd_agent_id
uint8[] cmd_label_idx        # into cmd_labels

# needs, one entry per (agent, need)
string[] need_names
int32[] need_agent_id
uint8[] need_name_idx        # into need_names
uint8[] need_slot            # position in the agent's need list
float32[] need_value         # 0..100

# interactions
int32[] interaction_id
string[] interaction_label
uint32[] interaction_offset  # into interaction_participants
int32[] interaction_participants

# level 2: observed agents, one entry per (observer, slot)
int32[] observed_agent_id
uint16[] observed_slot
float64[] observed_x
float64[] observed_y

# level 2: cached global paths
int32[] path_agent_id
uint32[] path_offset         # into path_x, path_y
float64[] path_x
float64[] path_y

# level 2: intermediate and final goals
int32[] igoal_agent_id
float64[] igoal_x
float64[] igoal_y
int32[] goal_agent_id
float64[] goal_x
float64[] goal_y
float64[] goal_theta

# level 2: waypoint lists
int32[] wp_agent_id
uint32[] wp_offset           # into wp_x, wp_y
float64[] wp_x
float64[] wp_y
uint32[] wp_active           # index of the active waypoint in the list
float64[] wp_active_radius   # arrival radius of the active waypoint, 0 = none
