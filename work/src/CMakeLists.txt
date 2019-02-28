
#########################################################
# Source Files
#########################################################

# ----TODO------------------- #
# list your source files here #
# --------------------------- #
SET(sources
	"patchmatch.hpp"
	"patchmatch.cpp"

	"opencv_util.hpp"
	"terrain.hpp"
	"terrain.cpp"

	"gain.hpp"
	"gain.cpp"

	"scott.hpp"
	"scott.cpp"

	"terrainopt.hpp"
	"terrainopt.cpp"
	
	"main.cpp"
	"CMakeLists.txt"
)

# Add executable target and link libraries
add_executable(${CGRA_PROJECT} ${sources})



#########################################################
# Link and Build Executable
#########################################################

# Set source groups (helper method)
target_source_group_tree(${CGRA_PROJECT})

# Link usage requirements
target_link_libraries(${CGRA_PROJECT} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${CGRA_PROJECT} PRIVATE tiff geotiff_library)