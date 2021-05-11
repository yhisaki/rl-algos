import random
import math
import wandb


# Start a new run
wandb.init(project='custom-charts')
offset = random.random()

# At each time step in the model training loop
for run_step in range(20):

  # Log basic experiment metrics, which show up as standard line plots in the UI
  wandb.log({
      "acc": math.log(1 + random.random() + run_step) + offset,
      "val_acc": math.log(1 + random.random() + run_step) + offset * random.random(),
  }, commit=False)

  # Set up data to log in custom charts
  data = []
  for i in range(100):
    data.append([i, random.random() + math.log(1 + i) + offset + random.random()])

  # Create a table with the columns to plot
  table = wandb.Table(data=data, columns=["step", "height"])

  # Use the table to populate various custom charts
  line_plot = wandb.plot.line(table, x='step', y='height', title='Line Plot')
  histogram = wandb.plot.histogram(table, value='height', title='Histogram')
  scatter = wandb.plot.scatter(table, x='step', y='height', title='Scatter Plot')

  # Log custom tables, which will show up in customizable charts in the UI
  wandb.log({'line_1': line_plot,
             'histogram_1': histogram,
             'scatter_1': scatter})
