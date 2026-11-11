"""Media Extract v2. Serialization matches real exported shortcuts:
text parameters (Text, URL Encode input, URL, notification body) are WFTextTokenString;
content parameters (Get Dictionary Value input) are WFTextTokenAttachment."""
import plistlib, uuid
BASE = "http://pufans-mac-mini.tail08155b.ts.net:8875/v1/inbox?url="
OBJ = "￼"
text_id, enc_id, get_id, msg_id = (str(uuid.uuid4()).upper() for _ in range(4))

def tokens(string, attachments):
    return {"Value": {"string": string, "attachmentsByRange": attachments}, "WFSerializationType": "WFTextTokenString"}

def out(action_id, name):
    return {"Type": "ActionOutput", "OutputUUID": action_id, "OutputName": name}

actions = [
    # 1. Everything that was shared (a URL item, or title + link text) as one piece of text.
    {"WFWorkflowActionIdentifier": "is.workflow.actions.gettext",
     "WFWorkflowActionParameters": {"UUID": text_id,
        "WFTextActionText": tokens(OBJ, {"{0, 1}": {"Type": "ExtensionInput"}})}},
    # 2. Percent-encode it for the query string.
    {"WFWorkflowActionIdentifier": "is.workflow.actions.urlencode",
     "WFWorkflowActionParameters": {"UUID": enc_id, "WFEncodeMode": "Encode",
        "WFInput": tokens(OBJ, {"{0, 1}": out(text_id, "Text")})}},
    # 3. GET /v1/inbox?url=… over Tailscale.
    {"WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
     "WFWorkflowActionParameters": {"UUID": get_id, "WFHTTPMethod": "GET", "ShowHeaders": False,
        "WFURL": tokens(BASE + OBJ, {f"{{{len(BASE)}, 1}}": out(enc_id, "URL Encoded Text")})}},
    # 4. The server always answers with a readable "message" (✅ queued / ❌ why not).
    {"WFWorkflowActionIdentifier": "is.workflow.actions.getvalueforkey",
     "WFWorkflowActionParameters": {"UUID": msg_id, "WFGetDictionaryValueType": "Value", "WFDictionaryKey": "message",
        "WFInput": {"Value": out(get_id, "Contents of URL"), "WFSerializationType": "WFTextTokenAttachment"}}},
    {"WFWorkflowActionIdentifier": "is.workflow.actions.notification",
     "WFWorkflowActionParameters": {"WFNotificationActionTitle": "Media Extract", "WFNotificationActionSound": False,
        "WFNotificationActionBody": tokens(OBJ, {"{0, 1}": out(msg_id, "Dictionary Value")})}},
]
shortcut = {
    "WFWorkflowActions": actions,
    "WFWorkflowClientVersion": "2607.0.2",
    "WFWorkflowMinimumClientVersion": 900,
    "WFWorkflowMinimumClientVersionString": "900",
    "WFWorkflowIcon": {"WFWorkflowIconStartColor": 4282601983, "WFWorkflowIconGlyphNumber": 59511},
    "WFWorkflowImportQuestions": [],
    "WFWorkflowTypes": ["ActionExtension"],
    "WFWorkflowInputContentItemClasses": ["WFURLContentItem", "WFStringContentItem", "WFRichTextContentItem", "WFSafariWebPageContentItem"],
    "WFWorkflowHasShortcutInputVariables": True,
    # Tapped from the Shortcuts app or a widget with no share input: use the clipboard.
    "WFWorkflowNoInputBehavior": {"Name": "WFWorkflowNoInputBehaviorGetClipboard", "Parameters": {}},
    "WFWorkflowOutputContentItemClasses": [],
    "WFQuickActionSurfaces": [],
}
import sys
with open(sys.argv[1] if len(sys.argv) > 1 else "Media Extract.unsigned.shortcut", "wb") as f:
    plistlib.dump(shortcut, f, fmt=plistlib.FMT_BINARY)
# Self-check: every text parameter is a token string, every content input an attachment.
for action in actions:
    for key, value in action["WFWorkflowActionParameters"].items():
        if isinstance(value, dict):
            kind = value["WFSerializationType"]
            expected = "WFTextTokenAttachment" if action["WFWorkflowActionIdentifier"].endswith("getvalueforkey") and key == "WFInput" else "WFTextTokenString"
            assert kind == expected, (action["WFWorkflowActionIdentifier"], key, kind)
            if kind == "WFTextTokenString":
                v = value["Value"]
                for rng in v["attachmentsByRange"]:
                    pos = int(rng.strip("{}").split(",")[0])
                    assert v["string"][pos] == OBJ, (key, rng)
print("built + self-check ok")
