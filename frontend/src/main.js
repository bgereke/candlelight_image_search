import Vue from "vue";
import App from "./App.vue";
import infiniteScroll from "vue-infinite-scroll";
import vuetify from './plugins/vuetify'
import VueMasonry from 'vue-masonry-css'

Vue.config.productionTip = false;

Vue.use(VueMasonry);
Vue.use(infiniteScroll);

new Vue({
  vuetify,
  render: h => h(App)
}).$mount("#app");
