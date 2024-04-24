import numpy as np
import matplotlib.pyplot as plt

# losses = np.array([1.5046289344088428, 8.10769784414516, 11.92065702536315, 14.398427557969262, 2.2216184300118225])
losses = np.array([2.2474730214083425, 8.292926355671186, 12.217682942162828, 14.5392328561973, 3.0344460245825853])
alpha = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# Plot the loss values for each alpha
plt.plot(alpha, losses, marker='o', linestyle='-', color='b')
plt.xlabel("Alpha (fraction of model 1)")
plt.ylabel("Loss")
plt.grid(True)
plt.title("Interpolation Loss")

# Save the plot
plt.savefig("interpolation_loss_pt.png")
