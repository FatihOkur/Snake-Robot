import matplotlib.pyplot as plt
from environment import DebrisMap

# 1. Initialize the new narrower/shorter map
env = DebrisMap(width=45, height=70, map_type="complex_map")

# 2. Setup the plot
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

# 3. Plot the Raw Map
axes[0].imshow(env.raw_grid, cmap='Greys', origin='lower')
axes[0].set_title("Raw Debris Map\n(Black = Wall/Debris, White = Free)")
axes[0].grid(color='cyan', linestyle='-', linewidth=0.2)

# 4. Plot the Inflated Planning Grid
axes[1].imshow(env.planning_grid, cmap='Reds', origin='lower')
axes[1].set_title("Inflated Planning Map\n(Dark Red = Danger Zone)")
axes[1].grid(color='cyan', linestyle='-', linewidth=0.2)
plt.tight_layout()

# 5. Save and display the image
plt.savefig("map_preview.png", dpi=300)
print("[OK] Saved map_preview.png successfully!")
plt.show()
