require("dotenv").config();

const csv = require("csv");
const fs = require("fs");
const { Facebook, FacebookApiException } = require("fb");

const filePath = "data/dv_dataset_consolidated.csv";

async function getPostIDs () {
  // let postIDs = ['431393286921843_1760033084057850']
  let postIDs = []
  
  await fs.createReadStream(filePath)
  .pipe(csv.parse())
  .on("data", (row) => {
    // console.log("parsing one row of CSV, row[1] is: " + row[1])
    postIDs.push(row[1])
  })
  .on("end", () => {
    console.log("Done reading " + filePath);
    // console.log("Returning postIDs: " + postIDs)
    return postIDs;
  });

}

async function get_access_token () {
  const fb = new Facebook({appId: process.env.FB_APP_ID, appSecret: process.env.FB_APP_SECRET});
  
  // console.log("about to call api, client_id is: " + process.env.FB_APP_ID)

  fb.api('oauth/access_token', {
      client_id: process.env.FB_APP_ID,
      client_secret: process.env.FB_APP_SECRET,
      grant_type: 'client_credentials'
  }, function (response) {
      if(!response || response.error) {
          console.log("[!] OAuth Error Occurred")
          console.log(!response ? `Response was ${response}` : response.error);
          return;
      } 
      const accessToken = response.access_token;
      console.log("Setting access token: " + accessToken)
      fb.setAccessToken(accessToken);
  });

  // console.log("returning fb object with access token:")
  // console.log(fb)
  return fb;

}

async function load_posts () {
  const postIDs = await getPostIDs()
  // console.log("in load_posts, post IDs is: " + postIDs);
  const fb = await get_access_token()
  await postIDs.forEach(postId => {
    fb.api(`/${postId}`, function (response) {
      if (response && !response.error) {
        console.log("(Success)")
        console.log(response);
      } else {
        console.log("[!] Getting FB Post Failed:")
        console.log(response.error);
      }
    });
  });
}

load_posts();



