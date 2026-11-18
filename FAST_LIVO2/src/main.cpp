#include "LIVMapper.h"
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  int result=0;
  try {
    auto node=std::make_shared<rclcpp::Node>("fast_livo");
    LIVMapper mapper(node);
    mapper.initializeSubscribersAndPublishers();
    mapper.run();
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "%s", error.what());
    result=1;
  }
  if (rclcpp::ok()) rclcpp::shutdown();
  return result;
}
