#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <arena_humansim_msgs/msg/agent_viz.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "arena_humansim_viz/build_markers.hpp"

namespace arena_humansim_viz
{

class VizNode : public rclcpp::Node
{
public:
  VizNode()
  : Node("arena_humansim_viz")
  {
    offset_x_ = declare_parameter("offset_x", 0.0);
    offset_y_ = declare_parameter("offset_y", 0.0);
    publisher_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(declare_parameter("output_topic",
        std::string("viz")), 100);
    parameters_ = add_post_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & parameters) {
        for (const auto & p : parameters) {
          if (p.get_name() == "offset_x") {
            offset_x_ = p.as_double();
          } else if (p.get_name() == "offset_y") {
            offset_y_ = p.as_double();
          } else if (p.get_name() == "output_topic") {
            publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(p.as_string(), 100);
          }
        }
      });
    watch_ = create_wall_timer(std::chrono::seconds(1), [this]() {follow_subscribers();});
  }

private:
  void follow_subscribers()
  {
    const bool watched = publisher_->get_subscription_count() > 0;
    if (watched && !subscription_) {
      subscription_ = create_subscription<arena_humansim_msgs::msg::AgentViz>(
        "viz_state", rclcpp::QoS(10).reliable().durability_volatile(),
        [this](arena_humansim_msgs::msg::AgentViz::ConstSharedPtr msg) {on_viz(*msg);});
    } else if (!watched && subscription_) {
      subscription_.reset();
      keys_[0].clear();
      keys_[1].clear();
    }
  }

  void on_viz(const arena_humansim_msgs::msg::AgentViz & msg)
  {
    const KeySet & previous = keys_[current_ ^ 1];
    build_markers(msg, previous, keys_[current_], markers_);
    current_ ^= 1;
    shift_markers(markers_, offset_x_, offset_y_);
    if (!markers_.markers.empty()) {
      publisher_->publish(markers_);
    }
  }

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
  rclcpp::Subscription<arena_humansim_msgs::msg::AgentViz>::SharedPtr subscription_;
  rclcpp::node_interfaces::PostSetParametersCallbackHandle::SharedPtr parameters_;
  rclcpp::TimerBase::SharedPtr watch_;
  double offset_x_ = 0.0;
  double offset_y_ = 0.0;
  std::array<KeySet, 2> keys_;
  size_t current_ = 0;
  visualization_msgs::msg::MarkerArray markers_;
};

}  // namespace arena_humansim_viz

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<arena_humansim_viz::VizNode>());
  rclcpp::shutdown();
  return 0;
}
