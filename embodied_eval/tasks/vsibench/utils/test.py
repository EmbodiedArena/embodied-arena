import json

# my_dict = {'name': 'Alice', 'age': 25, 'city': 'New York'}
# # my_jsonstr = json.dumps(my_dict)

# with open("/your/path/to/embodied-eval-main/logs/gpt5_2_res/res.txt","a") as f:
#     json.dump(my_dict, f)
#     f.write("\n-----------------------------------\n")

obj_appearance_order_accuracy = 0.4466019417475728
object_abs_distance_MRA = 0.2479616306954437
object_counting_MRA = 0.359646017699115
object_rel_distance_accuracy = 0.491549295774648
object_size_estimation_MRA = 0.386883525708289
room_size_estimation_MRA = 0.156944444444445
route_planning_accuracy = 0.3814432989690721

object_rel_direction_easy = 0.479262672811060
object_rel_direction_medium = 0.3809523809523809
object_rel_direction_hard = 0.2815013404825738

object_rel_direction_accuracy = (object_rel_direction_easy+object_rel_direction_medium+object_rel_direction_hard)/3.0

accuracy = (obj_appearance_order_accuracy+object_rel_distance_accuracy+route_planning_accuracy+object_rel_direction_accuracy)/4.0
mra = (object_abs_distance_MRA+object_counting_MRA+object_size_estimation_MRA+room_size_estimation_MRA)/4.0
overall = (accuracy+mra)/2

print("object_rel_direction_accuracy=" + str(object_rel_direction_accuracy))
print("accuracy=" + str(accuracy))
print("MRA=" + str(mra))
print("overall=" + str(overall))