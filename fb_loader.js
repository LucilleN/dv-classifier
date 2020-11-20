require("dotenv").config();
const csv = require("csv");
const fs = require("fs");
const { Facebook, FacebookApiException } = require("fb");

const filePath = "data/dv_dataset_consolidated.csv";
const fb = new Facebook({appId: process.env.FB_APP_ID, appSecret: process.env.FB_APP_SECRET});

fb.api('oauth/access_token', {
    client_id: process.env.FB_APP_ID,
    client_secret: process.env.FB_APP_SECRET,
    grant_type: 'client_credentials'
}, function (res) {
    if(!res || res.error) {
        console.log(!res ? 'error occurred' : res.error);
        return;
    } 
    console.log('\nSFRDUJWERUJHG\n')
    var accessToken = res.access_token;
    console.log(accessToken)
    fb.setAccessToken(accessToken);
});

fs.createReadStream(filePath)
  .pipe(csv.parse())
  .on("data", (row) => {
    label = row[0];
    postId = row[1];
    fb.api(`/${postId}`, function (response) {
      if (response && !response.error) {
        console.log(response);
      } else {
        console.log(response.error);
      }
    });
  })
  .on("end", () => {
    console.log("Done reading " + filePath);
  });
