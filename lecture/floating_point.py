from decimal import Decimal

PI = "110010010000111111011011"

s = Decimal(0)
for p, b in enumerate(PI):
    s += Decimal(int(b)) * Decimal(2)**(-Decimal(p))

print(s * 2)
