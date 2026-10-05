"""Synthetic support-ticket generator for Jev-Lite.
Gives full label control + clean calibration evaluation.
3 main heads (urgent, queue, severity) + 1 stretch null-type head (refund clarity).
"""
import random, csv, os

random.seed(42)

N = 3000

QUEUES = ["billing", "technical", "account", "other"]

SUBJECTS = {
    "billing": [
        "Incorrect charge on my invoice", "Double billed this month",
        "Refund not received", "Subscription renewal question",
        "Payment failed but money taken", "Need VAT invoice copy",
        "Charged after cancellation", "Plan upgrade billing issue",
    ],
    "technical": [
        "App crashes on login", "500 error on dashboard",
        "Feature not loading, stuck on spinner", "Sync failing across devices",
        "Slow performance since update", "Can't upload files, timeout error",
        "Notifications not arriving", "Integration webhook returning 403",
    ],
    "account": [
        "Can't reset my password", "Need to change account email",
        "How to close my account", "2FA code not working",
        "Locked out after too many attempts", "Update company name on account",
        "Transfer ownership to teammate", "Delete my data request",
    ],
    "other": [
        "Love the product, quick feedback", "Partnership inquiry",
        "Question about roadmap", "General question about pricing tiers",
        "Press inquiry", "Student discount question",
        "Feature suggestion: dark mode", "Hello, just testing support",
    ],
}

BODIES = {
    "billing": [
        "Hi, I was charged {amount} on {date} for invoice {inv}. This doesn't match my plan. Please check. {urg}",
        "Hello team, my card was charged twice for the same period. Transaction IDs {t1} and {t2}. Need a refund for one. {urg}",
        "I requested a refund {days} days ago for order {ord} and still haven't received it. {urg}",
        "Our subscription renewed at a higher price without notice. Can you explain the new charge of {amount}? {urg}",
    ],
    "technical": [
        "Hi, every time I {action} the app crashes. I'm on version {ver} with {os}. Steps: {steps} {urg}",
        "Getting a {code} error when opening {page}. Started {when}. I've cleared cache and retried. {urg}",
        "Since yesterday's update, {feature} takes over {secs}s to load and sometimes times out. {urg}",
        "Sync between web and mobile stopped working. Changes on {device} never appear. {urg}",
    ],
    "account": [
        "Hi, I tried to {action} but never got the email. Checked spam. My username is {user}. {urg}",
        "Need help with {action}. Account email is {email}. {urg}",
        "I'm locked out of {user} after enabling 2FA. Backup codes don't work. {urg}",
        "Please {action} for account {user}. Happy to verify identity. {urg}",
    ],
    "other": [
        "Hi! Just wanted to say {praise}. Also curious: {question} {urg}",
        "Hello, we're a {org} interested in {topic}. Who should we talk to? {urg}",
        "Quick question: {question} No rush. {urg}",
        "Sharing an idea: {idea} Thanks! {urg}",
    ],
}

URGENT_PHRASES = [
    "This is URGENT, production is down, need help ASAP!",
    "Please fix immediately, we're blocked and losing revenue.",
    "Critical: entire team can't work. Escalate please!",
    "ASAP please, deadline today.",
]
NONURGENT_PHRASES = [
    "No rush, whenever you get a chance.",
    "Just wondering when convenient.",
    "Low priority, thanks!",
    "Take your time, not blocking.",
]
NEUTRAL_PHRASES = ["Thanks!", "Let me know.", "Appreciate your help.", "Regards."]

REFUND_CLEAR_YES = [
    "Please refund order {ord} for {amount}.",
    "I want my money back for the duplicate charge.",
    "Refund requested: full amount to original card.",
]
REFUND_CLEAR_NO = [
    "Just checking my invoice details, no refund needed.",
    "Question about how billing cycles work.",
    "Can you explain the line items?",
]
REFUND_UNCLEAR = [
    "Not sure if I want a refund or credit, maybe?",
    "Might need money back, or maybe just fix the charge?",
    "Billing seems off... not sure what to ask for.",
]

PRIORITIES = ["low", "medium", "high", "critical"]
PRIO_TO_SEV = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}

def pick_urg_phrase(is_urgent):
    if is_urgent:
        return random.choice(URGENT_PHRASES + NEUTRAL_PHRASES[:1])
    else:
        # 10% of non-urgent still sound urgent-ish (label noise / hard cases)
        if random.random() < 0.10:
            return random.choice(URGENT_PHRASES)
        return random.choice(NONURGENT_PHRASES + NEUTRAL_PHRASES)

def gen_ticket(i):
    queue = random.choices(QUEUES, weights=[0.28, 0.32, 0.22, 0.18])[0]
    # priority correlates with queue but with noise
    if queue == "technical":
        prio = random.choices(PRIORITIES, weights=[0.15, 0.35, 0.35, 0.15])[0]
    elif queue == "billing":
        prio = random.choices(PRIORITIES, weights=[0.2, 0.4, 0.3, 0.1])[0]
    else:
        prio = random.choices(PRIORITIES, weights=[0.35, 0.4, 0.2, 0.05])[0]
    severity = PRIO_TO_SEV[prio] + random.gauss(0, 0.07)
    severity = float(min(1.0, max(0.05, severity)))
    is_urgent = 1 if (prio in ("high", "critical") and random.random() < 0.85) else (1 if random.random() < 0.08 else 0)

    subject = random.choice(SUBJECTS[queue])
    body_tpl = random.choice(BODIES[queue])
    urg = pick_urg_phrase(is_urgent)
    body = body_tpl.format(
        amount=f"${random.randint(9,499)}.{random.randint(10,99)}",
        date=f"2026-{random.randint(1,9):02d}-{random.randint(1,28):02d}",
        inv=f"INV-{random.randint(10000,99999)}",
        t1=f"txn_{random.randint(100000,999999)}", t2=f"txn_{random.randint(100000,999999)}",
        days=random.randint(2,21), ord=f"ORD-{random.randint(1000,9999)}",
        action=random.choice(["log in", "reset my password", "open settings", "export data", "invite a teammate"]),
        ver=f"{random.randint(2,5)}.{random.randint(0,9)}.{random.randint(0,9)}",
        os=random.choice(["Windows 11", "macOS 15", "Ubuntu 24.04", "iOS 18", "Android 15"]),
        steps="open app > click dashboard > crash",
        code=random.choice(["500", "502", "403", "429"]),
        page=random.choice(["dashboard", "reports page", "billing page", "settings"]),
        when=random.choice(["this morning", "after the update", "yesterday", "an hour ago"]),
        feature=random.choice(["search", "reports", "export", "dashboard"]),
        secs=random.randint(10, 120),
        device=random.choice(["mobile", "desktop"]),
        user=f"user{random.randint(100,9999)}",
        email=f"user{random.randint(100,9999)}@example.com",
        praise=random.choice(["love the new UI", "great support last time"]),
        question=random.choice(["do you offer SSO on team plan?", "is there an API for exports?", "when is the next release?"]),
        org=random.choice(["a university", "a startup", "a nonprofit"]),
        topic=random.choice(["an integration", "reselling", "a demo"]),
        idea=random.choice(["bulk export would help", "keyboard shortcuts please"]),
        urg=urg,
    )
    # refund clarity head (null-type demo): yes/no/unclear
    r = random.random()
    if queue == "billing" and random.random() < 0.6:
        if r < 0.45:
            refund = "yes"; body += " " + random.choice(REFUND_CLEAR_YES).format(ord=f"ORD-{random.randint(1000,9999)}", amount="$49.00")
        elif r < 0.8:
            refund = "no"; body += " " + random.choice(REFUND_CLEAR_NO)
        else:
            refund = "unclear"; body += " " + random.choice(REFUND_UNCLEAR)
    else:
        if r < 0.1:
            refund = "yes"; body += " " + random.choice(REFUND_CLEAR_YES).format(ord=f"ORD-{random.randint(1000,9999)}", amount="$19.00")
        elif r < 0.7:
            refund = "no"
        else:
            refund = "unclear"; body += " " + random.choice(REFUND_UNCLEAR) if random.random() < 0.3 else ""

    text = subject + ". " + body
    return {
        "id": i, "text": text, "subject": subject,
        "queue": queue, "is_urgent": is_urgent,
        "priority": prio, "severity": round(severity, 3),
        "refund_clear": refund,
    }

def main():
    os.makedirs("data", exist_ok=True)
    rows = [gen_ticket(i) for i in range(N)]
    # shuffle then 80/10/10
    random.shuffle(rows)
    n = len(rows)
    n_train, n_val = int(n*0.8), int(n*0.1)
    splits = (["train"]*n_train + ["val"]*n_val + ["test"]*(n-n_train-n_val))
    for r, s in zip(rows, splits):
        r["split"] = s
    with open("data/tickets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","text","subject","queue","is_urgent","priority","severity","refund_clear","split"])
        w.writeheader(); w.writerows(rows)
    # print stats
    from collections import Counter
    print(f"Wrote {len(rows)} tickets to data/tickets.csv")
    print("split:", Counter(r["split"] for r in rows))
    print("queue:", Counter(r["queue"] for r in rows))
    print("urgent rate:", sum(r["is_urgent"] for r in rows)/len(rows))
    print("refund:", Counter(r["refund_clear"] for r in rows))

if __name__ == "__main__":
    main()
