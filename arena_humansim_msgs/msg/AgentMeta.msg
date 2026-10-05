# Slow-changing attributes of the engine's own agents, latched, sent on change
std_msgs/Header header
string[] policies            # registered local planner names, by policy_idx
int32[] agent_id
string[] name                # empty = unnamed
string[] handedness          # l | r, empty = right
