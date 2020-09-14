<template>
  <div id="page">
    <div id="crop-search">
        <Cropper ref="cropper" id="image-to-crop" :src="image" :wheelResize="false"/>
        <div class="button-wrapper">
          <span id="load-button" class="button" @click="$refs.file.click()">
              <input type="file" ref="file" @change="uploadImage($event)" accept="image/*">
              Upload
          </span>
          <span id="google-button" class="button" @click="candlelight">
            <img src="./icons/google.png" height="40px"/>
          </span>
          <span id="bing-button" class="button" @click="candlelight">
            <img src="./icons/bing.png" height="50px"/>
          </span>
          <span id="candle-button" class="button" @click="candlelight">
            <img src="./icons/candle.png" height="40px"/>
          </span>
        </div>
    </div>

    <v-app >
      <div v-infinite-scroll="searchAllImages" infinite-scroll-distance="10" infinite-scroll-disabled=inf_scroll_disabled>
        <v-container fluid>
          <masonry :gutter="5" :cols="{default: 6, 1000: 4, 700: 3, 400: 2}">
              <v-card 
              id="card"
              outlined 
              :hover="true" 
              :href="item.pageURL" 
              target="_blank" 
              v-for="(item, index) in images" :key="index" 
              class="mt-2 mb-2">
                <v-img id="card-image" :src="item.largeImageURL"></v-img>
                <v-card-text class="py-0">{{item.tags}}</v-card-text>
              </v-card>
          </masonry>
          <h id="stop_loading" v-if="inf_scroll_disabled"> no more results </h>
        </v-container>                        
      </div>
    </v-app>
  </div>
</template>

<script>
import { requestsMixin } from "./mixins/requestsMixin";
import {Cropper} from 'vue-advanced-cropper';
const background = chrome.extension.getBackgroundPage()

export default {
  mixins: [requestsMixin],
  components: {
    Cropper
    },
  data() {
    return {
      form: {},
      page: 1,
      page_limit: 10,
      containerId: null,
      images: [],
      append_cnt: 0,
      image:
        background.screenshotUrl,
      num_results: 20,
      inf_scroll_disabled: false,
      searched: false
    };
  },

  methods: {

    async candlelight() {
    //   const isValid = await this.$refs.observer.validate();
    //   if (!isValid) {
    //     return;
    //   }
      this.inf_scroll_disabled = false;
      this.page = 1;
      if (this.searched){
        this.form.keyword = 'cat';
      } else {
        this.form.keyword = 'dog';
        this.searched = true;
      }
      
      await this.searchAllImages();
    },

    async loadMore() {
      this.inf_scroll_disabled = false;
      await this.searchAllImages();
      this.inf_scroll_disabled = true;
    },

    async searchAllImages() {
      if (!this.form.keyword) {
        return;
      }
      const response = await this.searchImages(this.form.keyword, this.page);
      if (this.page == 1) {
        this.images = response.data.hits;
      } else {
        this.images = this.images.concat(response.data.hits);
        console.log(this.images);
      }
      this.page++;
      if (this.page == this.page_limit){
        this.inf_scroll_disabled = true;
      }
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
					this.image = e.target.result;
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