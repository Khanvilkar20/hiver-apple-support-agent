INTENTS = {
    "software_update": "Problems installing, updating, or using iOS/macOS after an update.",
    "battery_power": "Battery draining quickly, overheating, charging, or power-related problems.",
    "hardware_device": "Physical device problems such as screen, camera, buttons, speakers, or other hardware.",
    "icloud_data": "iCloud, photos, backups, syncing, missing or lost personal data.",
    "account_security": "Apple ID, account access, passwords, phishing, scams, or security concerns.",
    "app_service": "Problems with Apple apps or services such as FaceTime, Music, Health, Mail, etc.",
    "device_access": "Locked out, passcode, activation, restore, or inability to access a device.",
    "purchase_tradein": "Purchasing products, trade-ins, upgrades, reservations, or related questions.",
    "general_troubleshooting": "A technical problem where the specific cause or category is unclear and troubleshooting is requested.",
    "other": "Requests that do not fit the other support categories.",
}
def format_intents_for_prompt() -> str:
    return "\n".join(
        f"- {name}: {description}"
        for name, description in INTENTS.items()
    )