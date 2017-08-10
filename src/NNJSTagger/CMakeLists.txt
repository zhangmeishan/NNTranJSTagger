
include_directories(
  model
)

add_executable(NNJSTagger NNJSTagger.cpp)
target_link_libraries(NNJSTagger ${LIBS})
