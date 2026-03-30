def classify_score(score, health):
    if health < 30 or score < -0.05:
        return "🔴 CRITICAL"
    elif health < 60 or score < -0.01:
        return "🟠 WARNING"
    else:
        return "🟢 NORMAL"