# Active gestures of the engine's own agents, latched, sent on change
std_msgs/Header header
int32[] agent_id             # owner of gestures[i]
Gesture[] gestures
