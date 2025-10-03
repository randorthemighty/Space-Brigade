import deepimpact
import os
import time


def run_scenario(file_suffix):
    # Start the timer
    start_time = time.time()

    earth = deepimpact.Planet()
    impact_file_name = f"impact_parameter_list_{file_suffix}.csv"
    impact_file_path = os.sep.join(
        (os.path.dirname(__file__), "..", "impact_parameter_lists", impact_file_name)
    )
    probability, population = deepimpact.impact_risk(
        earth, impact_file=impact_file_path
    )

    # Sort the probability Df the 'Probability' col in descending order
    probability_sorted = probability.sort_values(by="probability", ascending=False)

    # Prepare the output strings
    output_str1 = probability_sorted.head(20).to_string()
    output_str2 = (
        f"Total population affected for {file_suffix} parameter list: "
        + f"{population['mean']:,.0f} +/- {population['stdev']:,.0f}"
    )

    # Stop the timer
    end_time = time.time()

    # Calculate the runtime
    runtime = end_time - start_time
    output_str3 = f"Runtime of the code: {runtime:.2f} seconds"

    # Write the output to a .txt file
    output_file = f"output_{file_suffix}.txt"
    with open(output_file, "w") as file:
        file.write(output_str1 + "\n\n")
        file.write(output_str2 + "\n")
        file.write(output_str3 + "\n")

    print(f"Output for {file_suffix} written to {output_file}")


# # Run scenarios for 10, 100, and 1000
# for file_suffix in [10, 100, 1000]:
#     run_scenario(file_suffix)
run_scenario(10000)
