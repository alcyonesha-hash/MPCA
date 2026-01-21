# Google Sheets Integration Setup

## Step 1: Create Google Sheet

1. Go to [Google Sheets](https://sheets.google.com)
2. Create a new spreadsheet named "Survey Responses"
3. Create headers in Row 1:

```
A: Timestamp
B: ParticipantName
C: ParticipantEmail
D: AgeGroup
E: TechExperience
F: AttentionCheckPassed
G: TotalTimeMs
H: SetIndex
I: SetType
J: ComparisonType
K: FullPosition
L: RawQ1
M: RawQ2
N: RawQ3
O: RawQ4
P: Q1_HumanLike
Q: Q2_LessAnnoying
R: Q3_NaturalFlow
S: Q4_KeepInChat
T: PrefersFullQ1
U: PrefersFullQ2
V: PrefersFullQ3
W: PrefersFullQ4
X: TimeSpentMs
```

## Step 2: Create Google Apps Script

1. In your Google Sheet, go to **Extensions > Apps Script**
2. Replace the default code with:

```javascript
function doPost(e) {
  try {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    const data = JSON.parse(e.postData.contents);

    const participant = data.participant;
    const metadata = data.metadata;

    // Add a row for each response
    data.responses.forEach(response => {
      const row = [
        metadata.completedAt,
        participant.name,
        participant.email,
        participant.ageGroup,
        participant.techExperience,
        participant.attentionCheckPassed,
        metadata.totalTimeMs,
        response.setIndex,
        response.setType,
        response.comparisonType,
        response.fullPosition,
        response.rawQ1,
        response.rawQ2,
        response.rawQ3,
        response.rawQ4,
        response.q1_humanLike,
        response.q2_lessAnnoying,
        response.q3_naturalFlow,
        response.q4_keepInChat,
        response.prefersFullQ1,
        response.prefersFullQ2,
        response.prefersFullQ3,
        response.prefersFullQ4,
        response.timeSpentMs
      ];
      sheet.appendRow(row);
    });

    return ContentService
      .createTextOutput(JSON.stringify({status: 'success'}))
      .setMimeType(ContentService.MimeType.JSON);

  } catch (error) {
    return ContentService
      .createTextOutput(JSON.stringify({status: 'error', message: error.toString()}))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

// Allow GET for testing
function doGet(e) {
  return ContentService
    .createTextOutput("Survey endpoint is working!")
    .setMimeType(ContentService.MimeType.TEXT);
}
```

3. Save the project (Ctrl+S)

## Step 3: Deploy as Web App

1. Click **Deploy > New deployment**
2. Select type: **Web app**
3. Configure:
   - Description: "Survey Response Handler"
   - Execute as: **Me**
   - Who has access: **Anyone**
4. Click **Deploy**
5. Authorize the app when prompted
6. Copy the **Web app URL** (looks like: `https://script.google.com/macros/s/xxx/exec`)

## Step 4: Configure Survey Website

1. Open `index.html`
2. Add before `</body>`:

```html
<script>
  window.GOOGLE_SHEETS_WEBHOOK_URL = "YOUR_WEB_APP_URL_HERE";
</script>
```

Or create a `config.js` file:

```javascript
window.GOOGLE_SHEETS_WEBHOOK_URL = "https://script.google.com/macros/s/xxx/exec";
```

## Step 5: Test

1. Open the survey in a browser
2. Complete a test submission
3. Check your Google Sheet for the new row

## Troubleshooting

### CORS Issues
Google Apps Script automatically handles CORS for web apps with "Anyone" access.
The fetch uses `mode: 'no-cors'` which should work.

### No Data Appearing
1. Check the Apps Script execution logs: **View > Executions**
2. Verify the Web App URL is correct
3. Test with the GET endpoint first (visit the URL in browser)

### Data Backup
The survey app automatically downloads a JSON file as backup if Google Sheets submission fails.

## Alternative: Firebase (if Google Sheets doesn't work)

```javascript
// Firebase config
const firebaseConfig = {
  apiKey: "xxx",
  authDomain: "xxx.firebaseapp.com",
  projectId: "xxx",
  storageBucket: "xxx.appspot.com",
  messagingSenderId: "xxx",
  appId: "xxx"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const db = firebase.firestore();

// Submit data
db.collection("survey_responses").add(surveyData);
```
