import matplotlib.pyplot as plt

# przykładowe dane
mags = [2.5, 3.6, 4.2, 4.8, 5.3, 6.0]
risks = [0.05, 0.10, 0.35, 0.50, 0.65, 0.82]

plt.scatter(mags, risks)
plt.title("Wpływ maksymalnej magnitudy 72h na ryzyko sejsmiczne")
plt.xlabel("Maksymalna magnituda 72h")
plt.ylabel("Estymowane ryzyko (0–1)")
plt.grid(True)
plt.savefig("mag_vs_risk.png")
