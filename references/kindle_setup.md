# Kindle Email Delivery Setup

Step-by-step guide to configure email delivery to your Kindle device.

## Overview

Amazon Kindle devices can receive documents via email. This skill uses Gmail SMTP to send generated briefing PDFs directly to your Kindle.

## Step 1: Find Your Kindle Email Address

1. Go to [Amazon Content & Devices](https://www.amazon.com/hz/mycd/digital-console/contentlist)
2. Click **Preferences** tab
3. Scroll to **Personal Document Settings**
4. Under **Send-to-Kindle Email Settings**, find your Kindle email
   - Format: `username@kindle.com` or `username@kindle.[country]`
   - Example: `john_doe@kindle.com`

**Note**: Each Kindle device has a unique email address.

## Step 2: Approve Sender Email Address

Amazon only accepts documents from approved email addresses.

1. In **Personal Document Settings**, scroll to **Approved Personal Document E-mail List**
2. Click **Add a new approved e-mail address**
3. Enter your Gmail address (the one you'll use for `GMAIL_USER`)
4. Click **Add Address**

**Important**: The sender email must be approved, or Amazon will reject the documents.

## Step 3: Set Up Gmail App Password

Gmail requires app-specific passwords for SMTP access.

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable **2-Step Verification** (if not already enabled)
3. Go to **App passwords**
   - You may need to search for "app passwords" in settings
4. Select **Mail** and **Other (Custom name)**
5. Enter name: "Morning Briefing Skill"
6. Click **Generate**
7. Copy the 16-character password (format: `xxxx xxxx xxxx xxxx`)

**Security Notes**:
- This is NOT your regular Gmail password
- App passwords bypass 2FA for specific apps
- Keep this password secure
- You can revoke it anytime from Google Account settings

## Step 4: Configure Environment Variables

Set these environment variables in your shell:

```bash
export GMAIL_USER="your_email@gmail.com"
export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
```

Or add to your `.bashrc` / `.zshrc`:

```bash
# Morning Briefing Credentials
export GMAIL_USER="your_email@gmail.com"
export GMAIL_APP_PASSWORD="xxxxxxxxxxxxxxxx"
```

**Note**: Remove spaces from the app password if copying directly.

## Step 5: Update config.yaml

Edit your `config.yaml`:

```yaml
kindle_email: "your_username@kindle.com"
sender_email: "your_email@gmail.com"  # Must match GMAIL_USER
```

## Step 6: Test Email Delivery

Run a test:

```bash
python3 scripts/briefing_runner.py --config config.yaml --dry-run
```

This generates the PDF without sending. Then test email:

```bash
python3 scripts/kindle_sender.py \
  --pdf Personal-Briefing-Daily-2026.03.06.pdf \
  --kindle-email your_username@kindle.com
```

Check your Kindle device or Kindle app:
- Delivery usually takes 1-5 minutes
- Documents appear in your library
- Check "All" or "Documents" tab

## Troubleshooting

### Email Not Received

**Check 1: Approved Sender**
- Verify sender email is in your Approved Personal Document E-mail List
- Email addresses must match exactly (case-insensitive)

**Check 2: Gmail Credentials**
```bash
echo $GMAIL_USER
echo $GMAIL_APP_PASSWORD
```
Both should output valid values.

**Check 3: SMTP Authentication**
- Ensure 2-Step Verification is enabled on Google account
- App password must be generated from Google Account settings
- Try generating a new app password

**Check 4: Kindle Email Format**
- Must be `username@kindle.com` or `username@kindle.[country]`
- Check for typos

**Check 5: Amazon Settings**
- Personal Document Archiving must be enabled (default: on)
- Check your Amazon account's primary email for bounce notifications

### SMTP Authentication Failed

```
SMTPAuthenticationError: Username and Password not accepted
```

**Solutions**:
1. Verify `GMAIL_USER` matches the Gmail account exactly
2. Generate a new app password (old one may be revoked)
3. Ensure 2-Step Verification is enabled
4. Check for typos in app password (remove spaces)
5. Try logging into Gmail web to ensure account isn't locked

### Connection Timeout

```
TimeoutError: Connection timed out
```

**Solutions**:
1. Check internet connection
2. Verify firewall isn't blocking port 587
3. Try again (Gmail SMTP may be temporarily unavailable)

### File Size Limits

Amazon has document size limits:
- **Email attachment**: 50 MB max
- **Via Send to Kindle apps**: 200 MB max

If briefings exceed 50 MB:
- Reduce `max_papers`, `max_blogs`, `max_news` in config
- Decrease `arxiv_days_back` to fetch fewer papers
- Compress PDF (use external tools)

### Documents Not Syncing

If PDFs arrive but don't sync across devices:
1. Check **Personal Document Archiving** is enabled
2. Verify devices are registered to same Amazon account
3. Check device storage (Kindle may be full)
4. Manually sync: Settings → Sync and Check for Items

## Alternative Delivery Methods

If email delivery doesn't work, consider:

### USB Transfer
1. Generate PDF with `--dry-run` flag
2. Connect Kindle to computer via USB
3. Copy PDF to `documents` folder
4. Eject Kindle

### Send to Kindle App
1. Install [Send to Kindle desktop app](https://www.amazon.com/sendtokindle)
2. Right-click PDF → Send to Kindle
3. Select device and send

### Cloud Services
1. Upload PDF to cloud storage (Dropbox, Google Drive)
2. Open on mobile device
3. Use "Share" → Send to Kindle app

## Scheduling with Cron

For daily automatic delivery:

```bash
crontab -e
```

Add (runs every day at 7 AM):

```cron
0 7 * * * cd /path/to/morning-briefing && /usr/bin/python3 scripts/briefing_runner.py --config config.yaml >> logs/briefing.log 2>&1
```

**Tips**:
- Use absolute paths in cron jobs
- Environment variables may not be available in cron
- Consider using a wrapper script to source `.env` file

Example wrapper script (`run_briefing.sh`):

```bash
#!/bin/bash
cd /path/to/morning-briefing
source ~/.bashrc  # Load environment variables
python3 scripts/briefing_runner.py --config config.yaml >> logs/briefing.log 2>&1
```

Then in crontab:

```cron
0 7 * * * /path/to/morning-briefing/run_briefing.sh
```

## Privacy & Security

**Email Privacy**:
- PDFs are sent through Gmail's servers
- Amazon stores documents in your Personal Documents
- Consider data sensitivity when configuring topics

**API Keys**:
- Store in environment variables, never in code
- Use `.gitignore` to exclude `.env` files
- Rotate keys periodically
- Revoke unused app passwords

**Kindle Documents**:
- Amazon keeps documents in cloud unless deleted
- Documents sync across all your Kindle devices
- Manage at [amazon.com/mycd](https://www.amazon.com/hz/mycd/digital-console/contentlist)

## Additional Resources

- [Kindle Personal Documents Service](https://www.amazon.com/gp/help/customer/display.html?nodeId=GX9XLEVV8G4DB28H)
- [Gmail App Passwords](https://support.google.com/accounts/answer/185833)
- [Send to Kindle](https://www.amazon.com/sendtokindle)

## FAQ

**Q: Can I use a different email provider?**
A: Yes, but you'll need to modify `kindle_sender.py` to use different SMTP settings. Gmail is recommended for reliability.

**Q: Do I need a Kindle device?**
A: No, you can use the Kindle app on iOS, Android, Mac, or PC. Documents will sync to all devices.

**Q: How many documents can I send per day?**
A: Amazon doesn't publish a hard limit, but be reasonable. One briefing per day is fine.

**Q: Can I send to multiple Kindle devices?**
A: Yes, modify `config.yaml` to include multiple Kindle emails, or modify the script to send to a list.

**Q: Will this work with Kindle Paperwhite/Oasis/etc?**
A: Yes, all Kindle devices support personal documents. The `kindle` page format (6x8") is optimized for Kindle Scribe but works on all devices.

**Q: Can I delete documents from Kindle?**
A: Yes, swipe left on document and tap Delete, or manage at [amazon.com/mycd](https://www.amazon.com/hz/mycd/digital-console/contentlist).
