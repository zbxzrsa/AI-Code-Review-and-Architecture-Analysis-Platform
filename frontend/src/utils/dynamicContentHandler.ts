/*
 DynamicContentHandler
 - Hooks into fetch and element creation to ensure dynamic HTML gets translated
 - Designed to work with DOMTranslator
*/

// DOMTranslator 接口定义
interface DOMTranslator {
  translateEntirePage(): void;
  translateElement(el: Element): void;
}

export interface DynamicContentHandlerOptions {
  translator: DOMTranslator;
}

export class DynamicContentHandler {
  private translator: DOMTranslator;
  private restoreFns: Array<() => void> = [];

  constructor(options: DynamicContentHandlerOptions) {
    this.translator = options.translator;
  }

  init(): void {
    this.setupFetchHook();
    this.setupCreateElementHook();
  }

  destroy(): void {
    this.restoreFns.forEach((fn) => fn());
    this.restoreFns = [];
  }

  private setupFetchHook(): void {
    const originalFetch = window.fetch;
    const self = this;
    (window as any).fetch = async function (...args: any[]) {
      const response = await originalFetch.apply(this, args as any);
      try {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('text/html')) {
          // Schedule a page-wide translation pass after HTML affects DOM
          setTimeout(() => self.translator.translateEntirePage(), 0);
        }
      } catch (e) {
        console.warn('[DynamicContentHandler] fetch hook error', e);
      }
      return response;
    };
    this.restoreFns.push(() => { (window as any).fetch = originalFetch; });
  }

  private setupCreateElementHook(): void {
    const originalCreateElement = Document.prototype.createElement;
    const self = this;
    (Document.prototype as any).createElement = function (this: Document, tagName: string, options?: ElementCreationOptions) {
      const el = originalCreateElement.call(this, tagName, options);
      // Translate element after potential attributes/text are set asynchronously
      setTimeout(() => {
        try {
          self.translator.translateElement(el);
        } catch (e) {
          console.warn('[DynamicContentHandler] createElement hook error', e);
        }
      }, 0);
      return el;
    } as any;
    this.restoreFns.push(() => { (Document.prototype as any).createElement = originalCreateElement; });
  }
}

export default DynamicContentHandler;
