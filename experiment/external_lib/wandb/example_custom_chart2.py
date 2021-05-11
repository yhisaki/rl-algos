import math
import random
import wandb

# Start a new run
wandb.init(project='custom-charts',
                 notes='Custom stacked bar chart')
offset = random.random()

# Set up data to log in custom charts
data = []
for i in range(100):
  data.append([i, random.random() + math.log(1 + i) + offset + random.random()])

# Create a table with the columns to plot
table = wandb.Table(data=data, columns=["step", "height"])

# Map from the table's columns to the chart's fields
fields = {"x": "step",
          "value": "height"}

# Use the table to populate the new custom chart preset
# To use your own saved chart preset, change the vega_spec_name
my_custom_chart = wandb.plot_table(vega_spec_name="carey/new_chart",
              data_table=table,
              fields=fields,
              )

# Log the plot to have it show up in the UI
wandb.log({"custom_chart": my_custom_chart})
