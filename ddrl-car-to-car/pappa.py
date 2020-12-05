

pappa = ["Town" + (str(a) if a >= 10 else ("0"+str(a))) for a in range(1,8)]
print(pappa)

pappa = [f"Town{a if a >= 10 else f'0{a}'}" for a in range(1, 8)]
print(pappa)