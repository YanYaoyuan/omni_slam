# The exported fast_lio libraries link these PCL components by name. Find only
# them: a component-less find_package(PCL), which ament_export_dependencies
# would generate, also requires visualization/VTK, and cross-compilation
# sysroots such as RDK S100 do not provide a usable VTK.
find_package(PCL REQUIRED COMPONENTS common io filters kdtree registration)
