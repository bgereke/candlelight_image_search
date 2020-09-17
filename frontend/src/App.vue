<template>
  <div id="page">
    <div id="crop-search">
        <Cropper ref="cropper" id="image-to-crop" :src="background_image" :wheelResize="false"/>
        <div class="button-wrapper">
          <span id="load-button" class="button" @click="$refs.file.click()">
              <input type="file" ref="file" @change="uploadImage($event)" accept="image/*">
              Upload
          </span>
          <span id="google-button" class="button" @click="google_callback">
            <img src="./icons/google.png" height="40px"/>
          </span>
          <span id="bing-button" class="button" @click="candlelight_callback">
            <img src="./icons/bing.png" height="50px"/>
          </span>
          <span id="candle-button" class="button" @click="candlelight_callback">
            <img src="./icons/candle.png" height="40px"/>
          </span>
        </div>
    </div>

    <v-app >
      <div v-infinite-scroll="load_more" infinite-scroll-distance="10" infinite-scroll-disabled=inf_scroll_disabled>
        <v-container fluid>
          <masonry :gutter="5" :cols="{default: 6, 1000: 4, 700: 3, 400: 2}">
              <v-card 
              id="card"
              outlined 
              :hover="true" 
              v-for="(image, index) in images" :key="index"
              :href="image.page_url" 
              target="_blank"                
              class="mt-2 mb-2">
                <v-img id="card-image" :src="image.src_url"></v-img>
                <v-card-text class="py-0">{{image.keywords}}</v-card-text>
              </v-card>
          </masonry>
          <h id="stop_loading" v-if="inf_scroll_disabled"> no more results </h>
        </v-container>                        
      </div>
    </v-app>
  </div>
</template>

<script>
import {Cropper} from 'vue-advanced-cropper';
const axios = require("axios");
const APIURL = "https://pixabay.com/api";
const background = chrome.extension.getBackgroundPage();

export default {
  components: {
    Cropper
    },
  data() {
    return {
      form: {},
      loads: 1,
      load_limit: 5,
      containerId: null,
      images: [],
      append_cnt: 0,
      background_image:
        background.screenshotUrl,
      num_candlelight_results: 20,
      max_google_results: 50,
      inf_scroll_disabled: false,
      searched: false,
      service: '',
      token: ''
    };
  },

  methods: {

    async candlelight_callback() {
      this.images = [];
      this.service = 'candlelight';
      this.inf_scroll_disabled = false;
      this.loads = 1;      
      if (this.searched){
        this.form.keyword = 'cat';
      } else {
        this.form.keyword = 'dog';
        this.searched = true;
      }      
      await this.search_candlelight();
    },

    async google_callback() {
      this.images = [];
      this.service = 'google';
      this.inf_scroll_disabled = false;
      this.loads = 1;          
      this.search_google();
    },

    async load_more() {
      if (this.service == 'candlelight') {
        await this.search_candlelight();
      } else if (this.service == 'google') {
        await this.search_google();
      }
    },

    async search_candlelight() {
      if (!this.form.keyword) {
        return;
      }
      const response = await this.searchImages(this.form.keyword, this.loads);
      for (var i = 0; i < this.num_candlelight_results; i++){
        this.images.push({
          page_url: response.data.hits[i].pageURL,
          src_url: response.data.hits[i].largeImageURL,
          keywords: response.data.hits[i].tags
        });         
      }   
      this.loads++;
      if (this.loads == this.load_limit){
        this.inf_scroll_disabled = true;
      }
    },

    searchImages(keyword, page = 1) {
      return axios.get(
        `${APIURL}/?page=${page}&key=18137162-8bce742258e73e6063f58d40a&q=${keyword}`
      );
    },

    async search_google() {
      const cropped = this.$refs.cropper.getResult();        
      chrome.identity.getAuthToken({interactive: false}, function(token) {
        const background = chrome.extension.getBackgroundPage();
        background.token = token;
        }); 
      this.token = background.token;
      background.token = '';
      var b64_image = cropped.canvas.toDataURL('image/png').replace(/^data:image\/(png|jpg);base64,/, '');
      var request_body = {
        requests: [{
          image: {content: b64_image},
          features: [{
            type: 'WEB_DETECTION',
            maxResults: this.num_google_results
          }]
        }]
      };
      let request = {
        method: 'POST',
        async: true,
        headers: {
          Authorization: 'Bearer ' + this.token,
          'Content-Type': 'application/json'
        },
        contentType: 'json',
        body: JSON.stringify(request_body)
      };
      fetch(
        'https://vision.googleapis.com/v1/images:annotate?key=' + process.env.GOOGLE_KEY,
        request)
        .then(response => response.json())
        .then(data => {
          var response = data.responses[0].webDetection.visuallySimilarImages;  
          for (var i = 0; i < response.length; i++){
            this.images.push({
              page_url: response[i].url,
              src_url: response[i].url,
              keywords: ''
            });         
          }           
          this.inf_scroll_disabled = true;
        })
        .catch((error) => {
          console.error('Error:', error);
        });              
              
      //   axios({
      //     method: 'post',
      //     url: google_url,
      //     data: request_data
      //   })
      //   .then((response) => {
      //     console.log('response');
      //     console.log(response);
      //   }, (error) => {
      //     console.log('error');
      //     console.log(error);
      //   }); 
      // }, 'image/jpeg');       
    },

    uploadImage(event) {
      // Reference to the DOM input element
			var input = event.target;
			// Ensure that you have a file before attempting to read it
			if (input.files && input.files[0]) {
				// create a new FileReader to read this image and convert to base64 format
				var reader = new FileReader();
				// Define a callback function to run, when FileReader finishes its job
				reader.onload = (e) => {
					// Note: arrow function used here, so that "this.imageData" refers to the imageData of Vue component
					// Read image as base64 and set to imageData
					this.background_image = e.target.result;
				};
				// Start the reader job - read file as a data url (base64 format)
        reader.readAsDataURL(input.files[0]);
      }
    }
  }
};
</script>

<style scoped>
#image-to-crop {
  /* border: solid 1px #EEE; */
  height: 700px;
  width: 100%;
}

.button-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-top: 17px;
  /* margin-bottom:17px; */
}

.button {  
  cursor: pointer; 
  border-radius: 5px;
  margin-left: 17px;
  margin-right: 17px;
}

#load-button {
  font-family: Open Sans, Arial;
  color: white;
  font-size: 16px;
  font-weight: bold;
  padding: 9px 10px;
  text-align: center;
  background: #b3b3b3;
  transition: background 0.5s; 
}

.button:hover {opacity: 0.5}

.button input {
  display: none;
}

#card-image {
  border-radius: 10px;
}

#card {
  border-color: white;
  border-radius: 10px;
}

#stop_loading {
  display: flex;
  font-family: Open Sans, Arial;
  font-size: 32px;
  margin: 17px;
  text-align: center;
  justify-content: center;
}

</style>