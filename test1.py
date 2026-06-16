import matplotlib.pyplot as plt

years=[2018,2019,2020,2021,2022,2023,2024]
scale=[100,130,170,220,280,350,430]

plt.figure(figsize=(8,5))
plt.plot(years,scale,marker='o')
plt.xlabel("Year")
plt.ylabel("Market Scale(Billion)")
plt.title("Cloud Computing Market Growth")
plt.grid(True)

plt.savefig("cloud_market.png",dpi=300)
plt.show()