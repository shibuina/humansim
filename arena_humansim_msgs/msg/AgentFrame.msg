# Per-tick state of the engine's own agents, one array entry per agent
std_msgs/Header header
int32[] agent_id
float64[] x
float64[] y
float64[] theta
float64[] vx
float64[] vy
float64[] desired_velocity   # preferred walking speed (m/s)
float64[] radius             # collision radius (m)
uint8[] kind                 # AgentState.KIND_*
uint8[] animation_state
int16[] policy_idx           # into AgentMeta.policies, -1 = teleport-only
